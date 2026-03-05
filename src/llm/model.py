"""
LLM Configuration and Initialization for Trecl mult-agent system.
Currently configured to use Cerebras explicitly.
"""

from langchain_cerebras import ChatCerebras
from pydantic import SecretStr

from core.config import CEREBRAS_API_KEYS, USE_MOCK_LLM

class MockLLMResponse:
    def __init__(self, content: str):
        self.content = content

class MockStructuredRunner:
    """Mocks the LangChain Runnable returned by with_structured_output."""
    def __init__(self, schema):
        self.schema = schema

    def invoke(self, messages):
        # Returns an instantiated Pydantic object directly
        return self.schema(
            pain_points_ranked="1. Kubernetes resource scaling costs are too high\n2. PostgreSQL queries are slow due to lack of indexing\n3. Deployment pipelines are flaky",
            project_ideas="Project: Auto-Scaling K8s Operator in Go\nDetails: Build a custom Kubernetes operator using Go that monitors traffic spikes and dynamically provisions exact node resources, reducing Zepto's cloud spend while ensuring 100% uptime during peak delivery hours."
        )

class MockLLM:
    """A proxy LLM class used for local testing to avoid API limits and costs."""
    
    def with_structured_output(self, schema):
        return MockStructuredRunner(schema)
        
    def invoke(self, messages) -> MockLLMResponse:
        """Mocks the LangChain execution, returning fixed outputs depending on context."""
        
        # Check standard contextual hints in what was sent to LLM to return relevant fake data
        full_text = str(messages)
        
        if "5 points strictly" in full_text:
            return MockLLMResponse(
                "- What they do: Zepto is a rapid grocery delivery startup operating via dark stores.\n"
                "- Tech stack: Heavy usage of PostgreSQL, Kubernetes, Backstage, and Argo.\n"
                "- Funding: Raised $450m Series J in Oct 2025 at $7B valuation.\n"
                "- Size: 1,000+ dark stores in 127 urban markets.\n"
                "- Challenge: Bad tech debt making infrastructure scaling difficult."
            )
        
        if "extract their core needs" in full_text or "Open Engineering Roles" in full_text:
            return MockLLMResponse(
                "- Open Roles: Senior Backend Engineer, Data Engineer\n"
                "- Requirements: Heavy Python/Go, PostgreSQL optimization, Kubernetes scaling.\n"
                "- Inferred Pain Points: Infrastructure scaling is severely lagging behind traffic growth, and data pipelines are bottlenecked."
            )
            
        if "deduce the company's top 3 technical pain points" in full_text:
            return MockLLMResponse(
                """{
  "pain_points_ranked": "1. Kubernetes resource scaling costs are too high\\n2. PostgreSQL queries are slow due to lack of indexing\\n3. Deployment pipelines are flaky",
  "project_ideas": "Project: Auto-Scaling K8s Operator in Go\\nDetails: Build a custom Kubernetes operator using Go that monitors traffic spikes and dynamically provisions exact node resources, reducing Zepto's cloud spend while ensuring 100% uptime during peak delivery hours."
}"""
            )
            
        if "Write a cold email" in full_text:
            return MockLLMResponse(
                "SUBJECT: Helping Zepto scale PostgreSQL\n\n"
                "Hi Zepto Team,\n\n"
                "I saw you are utilizing PostgreSQL and Argo to manage your 1000+ dark store network!\n"
                "Given your extremely rapid $450m-backed scaling effort, I imagine tech-debt and infrastructure challenges are a top priority. I'd love to help by implementing automated scalable architecture fixes as an intern.\n\n"
                "Thanks,\n[My Name]"
            )
            
        return MockLLMResponse("Mocked Response generated safely.")

        return MockLLMResponse("Mocked Response generated safely.")


# --- Retry Wrapper for the Production LLM ---

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_base,
    before_sleep_log
)
import logging

# Set up a basic logger to show when a retry happens
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Trecl.LLM.Retry")

def is_transient_llm_error(exception: Exception) -> bool:
    """
    Determines if an exception is a transient API/queue error (429/5xx).
    We DO NOT want to retry on 400s (Bad Request) or Pydantic ValidationErrors,
    because retrying a malformed prompt will just fail again and burn time.
    """
    err_str = str(exception).lower()
    
    # Catch the specific Cerebras queue error we saw in the logs
    if "429" in err_str or "too_many_requests_error" in err_str or "queue_exceeded" in err_str:
        return True
        
    # Catch standard generic server errors
    if "500" in err_str or "502" in err_str or "503" in err_str or "504" in err_str:
        return True
        
    return False

class retry_if_transient_llm_error(retry_base):
    """Custom tenacity retry strategy that checks our specific error criteria."""
    def __call__(self, retry_state):
        if retry_state.outcome.failed:
            return is_transient_llm_error(retry_state.outcome.exception())
        return False

class ChatCerebrasWithRetry:
    """
    A proxy class that wraps the underlying LangChain ChatCerebras instance.
    Applying @retry here automatically protects ALL agents in the pipeline
    that call invoke() or with_structured_output().
    """
    def __init__(self, llm_instance):
        self._llm = llm_instance
        
    @retry(
        # Wait 2^x * 1 seconds between each retry, maxing out at 10 seconds wait
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
        # Only retry if our helper function says it's a transient 429/5xx
        retry=retry_if_transient_llm_error(),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def invoke(self, *args, **kwargs):
        return self._llm.invoke(*args, **kwargs)

    def with_structured_output(self, schema):
        # with_structured_output returns a new Runnable, so we must wrap THAT runnable too
        # so its invoke() method is protected by the same retry logic.
        structured_runner = self._llm.with_structured_output(schema)
        
        class StructuredRunnerWithRetry:
            @retry(
                wait=wait_exponential(multiplier=1, min=2, max=10),
                stop=stop_after_attempt(5),
                retry=retry_if_transient_llm_error(),
                before_sleep=before_sleep_log(logger, logging.WARNING)
            )
            def invoke(self, *args, **kwargs):
                return structured_runner.invoke(*args, **kwargs)
                        
        return StructuredRunnerWithRetry()
        
    def bind_tools(self, *args, **kwargs):
        # bind_tools also returns a new Runnable. We wrap the result so its invoke() is protected.
        bound_llm = self._llm.bind_tools(*args, **kwargs)
        return ChatCerebrasWithRetry(bound_llm)


def get_llm():
    """
    Initializes and returns the central Language Model instance.
    Utilizes a round-robin API key manager for load balancing.
    
    Returns:
        ChatCerebrasWithRetry or MockLLM: Authenticated, retry-protected instance.
    """
    if USE_MOCK_LLM:
        print("\n[MOCK MODE] Skipping Cerebras API calls to avoid rate limits!")
        return MockLLM()
        
    base_llm = ChatCerebras(
        model="gpt-oss-120b",
        api_key=SecretStr(CEREBRAS_API_KEYS.get_next_key()),
        max_tokens=8192,
    )
    
    # Wrap the base LLM in our retry proxy
    return ChatCerebrasWithRetry(base_llm)

# Note: We keep this as a function to intentionally get a fresh key per initialization
llm = get_llm()
