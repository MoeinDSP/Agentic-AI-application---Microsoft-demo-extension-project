from pydantic import BaseModel, Field


class AgentSkill(BaseModel):
    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    tags: list[str] = Field(default_factory=list)


class AgentEndpoint(BaseModel):
    path: str = Field(min_length=1)
    method: str = Field(min_length=1)
    description: str = Field(min_length=1)


class AgentAuth(BaseModel):
    mode: str = Field(min_length=1)
    notes: str = Field(min_length=1)


class AgentCard(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    version: str = Field(min_length=1)
    base_url: str = Field(min_length=1)
    interaction_mode: str = Field(min_length=1)
    endpoints: list[AgentEndpoint] = Field(min_length=1)
    skills: list[AgentSkill] = Field(min_length=1)
    content_types: list[str] = Field(min_length=1)
    auth: AgentAuth
    notes: list[str] = Field(default_factory=list)
