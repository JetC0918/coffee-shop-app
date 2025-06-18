from typing import Protocol, List, Dict, Any


class AgentProtocol(Protocol):
    def get_response(self, messages:List[Dict[str,any]]) -> Dict[str,Any]:
        ...