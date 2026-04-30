from pydantic import BaseModel, Field, field_validator
from enum import Enum
from typing import Optional


class ToneEnum(str, Enum):
    professional = "Professional"
    conversational = "Conversational"
    luxury = "Luxury"
    enthusiastic = "Enthusiastic"
    minimalist = "Minimalist"
    technical = "Technical"
    playful = "Playful"
    empathetic = "Empathetic"


class CategoryEnum(str, Enum):
    consumer_electronics = "Consumer Electronics"
    health_wellness = "Health & Wellness"
    fashion_apparel = "Fashion & Apparel"
    home_living = "Home & Living"
    food_beverage = "Food & Beverage"
    software_saas = "Software / SaaS"
    beauty_skincare = "Beauty & Skincare"
    sports_fitness = "Sports & Fitness"
    automotive = "Automotive"
    b2b_enterprise = "B2B / Enterprise"
    other = "Other"


# class ResearchRequest(BaseModel):
#     product_name: str = Field(..., min_length=2, max_length=150, examples=["NovaPro Wireless Earbuds"])
#     category: CategoryEnum = Field(..., examples=["Consumer Electronics"])
#     key_features: str = Field(..., min_length=10, max_length=1000, examples=["40hr battery, ANC, IPX5 waterproof"])
#     target_audience: str = Field(..., min_length=5, max_length=300, examples=["Remote workers aged 25-40"])
#     tone: ToneEnum = Field(default=ToneEnum.professional)

#     @field_validator("product_name", "key_features", "target_audience")
#     @classmethod
#     def strip_whitespace(cls, v: str) -> str:
#         return v.strip()


# Tool result models
class WebSearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    source: str


class WebFetchResult(BaseModel):
    url: str
    content: str
    word_count: int


# Research output models
class MarketIntelligence(BaseModel):
    market_size_context: str
    growth_trend: str
    key_players: list[str]
    search_sources: list[str]


class AudienceInsights(BaseModel):
    top_pain_points: list[str]
    buying_motivations: list[str]
    language_patterns: list[str]
    search_sources: list[str]


class CompetitorAnalysis(BaseModel):
    top_competitors: list[str]
    common_messaging: list[str]
    gaps_and_opportunities: list[str]
    search_sources: list[str]


class ResearchSummary(BaseModel):
    market_intelligence: MarketIntelligence
    audience_insights: AudienceInsights
    competitor_analysis: CompetitorAnalysis
    recommended_positioning: str
    key_differentiators: list[str]
    content_angles: list[str]


class ToolCall(BaseModel):
    tool_name: str
    query: str
    status: str
    results_count: int


class ResearchResponse(BaseModel):
    success: bool
    product_name: str
    category: str
    research_summary: ResearchSummary
    tool_calls_made: list[ToolCall]
    total_sources_analyzed: int
    model_used: str
    usage: Optional[dict] = None


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: Optional[str] = None