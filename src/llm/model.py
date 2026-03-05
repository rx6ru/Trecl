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

def get_llm():
    """
    Initializes and returns the central Language Model instance.
    Utilizes a round-robin API key manager for load balancing.
    
    Returns:
        ChatCerebras or MockLLM: Authenticated instance of the model.
    """
    if USE_MOCK_LLM:
        print("\n[MOCK MODE] Skipping Cerebras API calls to avoid rate limits!")
        return MockLLM()
        
    return ChatCerebras(
        model="gpt-oss-120b",
        api_key=SecretStr(CEREBRAS_API_KEYS.get_next_key()),
        max_tokens=8192,
    )

# Note: We keep this as a function to intentionally get a fresh key per initialization
llm = get_llm()
