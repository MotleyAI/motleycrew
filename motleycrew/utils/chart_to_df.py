from typing import List

import pandas as pd
from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel, Field, model_validator

from gslides_api.domain import ImageData
from motleycrew.utils.structured_output_with_retries import structured_output_with_retries
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
        if x_axis_length == 0:
            raise ValueError("x_axis_values must have at least one value")
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


def extract_chart_data(
    image: str | ImageData, llm: BaseLanguageModel, series_names: List[str] | None = None
) -> ChartDataResult:
    """Extract chart data from image"""
    # Second call: Extract data points with retries
    print("\nExtracting chart data...")
    if series_names:
        series_names = ", ".join(series_names)
        series_names_instruction = f"""MAKE SURE the series names match the following: {series_names},
not necesarily in the same order (use the chart to match these names to the available series)"""
    else:
        series_names_instruction = """
        Take the series names from the legend if there is a legend. 
        If there is no legend and there is only one series, use the y-axis label if available. 
        In other cases, infer a meaningful series name from the chart overall.
        """

    data_prompt = f"""You are analyzing a chart image to extract precise data points.

    Reading methodology for bar and line charts:
    1. Identify x-axis labels/dates from left to right
    2. For each series, trace the line/bars and read y-values using grid lines as guides
    3. Align data points carefully with x-axis positions
    
    Reading methodology for pie and funnel charts:
    1. Identify the labels from the chart. Treat the labels as x axis values. If any of the pie slices lack labels, make some up. 
    2. The series values for a pie chart are the percentage of the pie slice corresponding to each label, 
    unless explicitly labeled otherwise.
    3. The series values for a funnel chart are the width of the funnel at each stage, if not explicitly labeled. 
    
    If the data points are only partially labeled, try to infer the missing values from the chart, while using 
    the labels you do have as much as possible. Consider that if there are apparently pointless
    numbers elsewhere in the image, they may be misplaced labels of the data points, at least
    for funnel and chart types. 

    For series_data: Create SeriesData objects with:
    - name: {series_names_instruction}
    - values: Numeric values as floats

    For x_axis_values: Extract all x-axis labels as strings, preserving original format.

    Data formatting:
    - Convert percentages to decimals (50% → 0.5)
    - Remove currency symbols but preserve numeric values
    - Use consistent decimal precision (2-3 decimal places)

    Validation:
    - Each series must have same number of values as x_axis_values
    - The series must not be empty
    - If a value is unclear, estimate based on nearby grid lines
    
    If the image is low-resolution, do your best to read the labels and values, 
    don't just return an empty list or ask for a better image. 

    Use the StructuredPassthroughTool to provide the result."""

    data_result = structured_output_with_retries(
        schema=ChartDataResult,
        prompt=data_prompt,
        input_messages=[image_to_human_message(image)],
        language_model=llm,
    )

    return data_result
