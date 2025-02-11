from typing import Annotated, List, Tuple, TypedDict, Union, Optional
from pydantic import BaseModel, Field

# Base model for plan
class Plan(BaseModel):
    """Plan to follow for query execution"""
    steps: List[str] = Field(description='Ordered execution steps')

# Base model for generating the response
class Response(BaseModel):
    """Reponse to the user"""
    response: str

# Base model for next action
class Act(BaseModel):
    """Action to perform"""
    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. \
                     If you need to further use tools to get the answer, user Plan."
    )

class DetectAmbiguity(BaseModel):
    """Schema for detecting ambiguity in a query."""
    is_ambiguous: bool  # Indicates whether the query is ambiguous
    follow_up: Optional[str] = None

class RefinedQuery(BaseModel):
    refined_query: str
