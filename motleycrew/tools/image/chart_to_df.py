from typing import List

import pandas as pd
from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel, Field, model_validator

from gslides_api.domain import ImageData
from motleycrew.agents.structured_output_with_retries import structured_output_with_retries
from motleycrew.tools.image import image_to_human_message


class SeriesData(BaseModel):
    name: str
    values: List[float]


class ChartDataResult(BaseModel):
    """Data points extracted from chart image"""

    series_data: List[SeriesData] = Field(description="Data for each series, keyed by series name")
    x_axis_values: List[str] = Field(description="X-axis values (time series or categories)")

    @model_validator(mode="after")
    def validate_series_data_length(self):
        """Validate that all series have the same length as x_axis_values"""
        x_axis_length = len(self.x_axis_values)
        for series in self.series_data:
            if len(series.values) != x_axis_length:
                raise ValueError(
                    f"Series '{series.name}' has {len(series.values)} values, "
                    f"but x_axis_values has {x_axis_length} values. "
                    f"All series must have the same length as x_axis_values."
                )
        return self

    def to_df(self, x_axis_name: str = "x_axis") -> pd.DataFrame:
        """Convert to PydanticSerializableDataFrame"""
        df_data = {x_axis_name: self.x_axis_values}
        for series in self.series_data:
            df_data[series.name] = series.values
        return pd.DataFrame(df_data)


def extract_chart_data(image: str | ImageData, llm: BaseLanguageModel) -> ChartDataResult:
    """Extract chart data from image"""
    # Second call: Extract data points with retries
    print("\nExtracting chart data...")

    data_prompt = """You are analyzing a chart image to extract precise data points.

    Reading methodology:
    1. Identify x-axis labels/dates from left to right
    2. For each series, trace the line/bars and read y-values using grid lines as guides
    3. Align data points carefully with x-axis positions

    For series_data: Create SeriesData objects with:
    - name: Must exactly match series names from legend
    - values: Numeric values as floats

    For x_axis_values: Extract all x-axis labels as strings, preserving original format.

    Data formatting:
    - Convert percentages to decimals (50% → 0.5)
    - Remove currency symbols but preserve numeric values
    - Use consistent decimal precision (2-3 decimal places)

    Validation:
    - Each series must have same number of values as x_axis_values
    - If a value is unclear, estimate based on nearby grid lines

    Use the StructuredPassthroughTool to provide the result."""

    data_result = structured_output_with_retries(
        schema=ChartDataResult,
        prompt=data_prompt,
        input_messages=[image_to_human_message(image)],
        language_model=llm,
    )

    return data_result
