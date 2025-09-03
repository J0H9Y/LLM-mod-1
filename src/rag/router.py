from typing import Dict, Any, Optional, List
import re
from dataclasses import dataclass
from enum import Enum, auto
from loguru import logger

class QueryIntent(Enum):
    """Supported query intents for the RAG pipeline."""
    GENERAL_KNOWLEDGE = auto()
    DATA_QUERY = auto()
    CODE_GENERATION = auto()
    DOCUMENT_LOOKUP = auto()
    MATH_CALCULATION = auto()
    OTHER = auto()

@dataclass
class RoutingResult:
    """Result of query routing."""
    intent: QueryIntent
    confidence: float
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.name,
            "confidence": self.confidence,
            "metadata": self.metadata or {}
        }

class QueryRouter:
    """
    Routes incoming queries to appropriate handlers based on intent classification.
    Uses a combination of rule-based and LLM-based classification.
    """
    
    def __init__(self, llm_client=None, rules: Optional[Dict] = None):
        """
        Initialize the query router.
        
        Args:
            llm_client: Optional LLM client for intent classification
            rules: Optional custom routing rules
        """
        self.llm = llm_client
        self.rules = rules or self._get_default_rules()
        self._setup_intent_patterns()
    
    def _get_default_rules(self) -> Dict:
        """Get default routing rules."""
        return {
            "intent_threshold": 0.7,  # Minimum confidence to accept LLM classification
            "cache_size": 1000,  # Number of queries to cache
            "timeout_seconds": 5,  # Max time to wait for LLM classification
        }
    
    def _setup_intent_patterns(self):
        """Initialize regex patterns for intent classification."""
        self.patterns = {
            QueryIntent.DATA_QUERY: [
                r"(show|list|get|find|retrieve).*data",
                r"(query|search).*database",
                r"(how many|how much|what is the).*(in|from|for)",
            ],
            QueryIntent.CODE_GENERATION: [
                r"(write|create|generate|show me).*code",
                r"(how to|how do I).*in (python|javascript|java|c\+\+|go|rust)",
                r"(function|script|program).*(that|which)",
            ],
            QueryIntent.MATH_CALCULATION: [
                r"(calculate|what is|how much is|solve).*[0-9+\-*/^()]+",
                r"(sum|add|multiply|divide|subtract)",
                r"(equation|formula|expression)",
            ],
            QueryIntent.DOCUMENT_LOOKUP: [
                r"(find|search|look up).*(document|file|pdf|docx|txt|markdown|md)",
                r"(what does).*(say|state|mention|indicate)",
                r"(summarize|extract).*from (the )?(document|file|text)",
            ]
        }
        
        # Compile all regex patterns
        for intent, patterns in self.patterns.items():
            self.patterns[intent] = [re.compile(p, re.IGNORECASE) for p in patterns]
    
    def classify_intent(self, query: str) -> RoutingResult:
        """
        Classify the intent of a query using rule-based and LLM-based methods.
        
        Args:
            query: The user's query string
            
        Returns:
            RoutingResult containing the detected intent and confidence
        """
        # First try rule-based classification (fast)
        rule_based = self._classify_with_rules(query)
        if rule_based.confidence > 0.8:  # High confidence from rules
            return rule_based
        
        # Fall back to LLM classification if available
        if self.llm:
            try:
                llm_result = self._classify_with_llm(query)
                # If LLM is more confident or rules had low confidence, use LLM result
                if llm_result.confidence > rule_based.confidence:
                    return llm_result
            except Exception as e:
                logger.warning(f"LLM classification failed: {e}")
        
        # Default to rule-based result or OTHER if no match
        return rule_based if rule_based.intent != QueryIntent.OTHER else RoutingResult(
            intent=QueryIntent.OTHER,
            confidence=1.0,
            metadata={"method": "fallback"}
        )
    
    def _classify_with_rules(self, query: str) -> RoutingResult:
        """Classify query intent using rule-based patterns."""
        best_match = None
        best_score = 0.0
        
        for intent, patterns in self.patterns.items():
            # Check if any pattern matches
            for pattern in patterns:
                if pattern.search(query):
                    # Simple scoring: count matching patterns
                    score = sum(1 for p in patterns if p.search(query)) / len(patterns)
                    if score > best_score:
                        best_score = score
                        best_match = intent
        
        if best_match and best_score > 0:
            return RoutingResult(
                intent=best_match,
                confidence=min(best_score, 0.9),  # Cap rule-based confidence
                metadata={"method": "rule_based"}
            )
        
        return RoutingResult(
            intent=QueryIntent.OTHER,
            confidence=0.5,
            metadata={"method": "rule_based"}
        )
    
    def _classify_with_llm(self, query: str) -> RoutingResult:
        """Classify query intent using the LLM."""
        if not self.llm:
            raise ValueError("LLM client not available for classification")
        
        # Prepare the prompt for intent classification
        system_prompt = """You are an intent classification system. Analyze the user's query and determine the most likely intent.
        
        Available intents:
        - GENERAL_KNOWLEDGE: General questions or information requests
        - DATA_QUERY: Requests to retrieve or analyze data
        - CODE_GENERATION: Requests to generate or explain code
        - DOCUMENT_LOOKUP: Requests to find or analyze documents
        - MATH_CALCULATION: Requests involving mathematical calculations
        - OTHER: Anything that doesn't fit the above categories
        
        Respond with ONLY the intent name in ALL_CAPS."""
        
        # Get the LLM response
        response = self.llm.query(
            prompt=query,
            system_prompt=system_prompt
        )
        
        # Parse the response
        intent_name = response.get("response", "").strip().upper()
        try:
            intent = QueryIntent[intent_name]
            confidence = 0.9  # High confidence for LLM classification
        except (KeyError, ValueError):
            intent = QueryIntent.OTHER
            confidence = 0.7  # Lower confidence if intent not recognized
        
        return RoutingResult(
            intent=intent,
            confidence=confidence,
            metadata={
                "method": "llm",
                "llm_response": response
            }
        )
