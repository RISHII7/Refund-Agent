from pydantic import BaseModel, Field

class Answer(BaseModel):
    answer: str = Field(description="The final response message to show the customer once all required actions and tool calls are finished. Must include the refund amount if a refund was issued.")
    conversation: bool = Field(
        description="Set to False when the customer's request is resolved (refund ticket issued, refund denied, or purchase history provided). "
        "Set to True ONLY if the customer gave an invalid username and you must ask for their username again.",
        default=False
    )

class PurchaseItem(BaseModel):
    name: str = Field(description="name of the item purchased")
