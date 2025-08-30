from motleycrew.common import LLMFramework
from motleycrew.common.llms import init_llm
from motleycrew.tools.image.image_utils import is_this_a_chart

llm = init_llm(llm_framework=LLMFramework.LANGCHAIN, llm_name="gpt-5")

girl_result = is_this_a_chart("images/girl.png", llm)
print(f"Is this a chart? {girl_result}")

chart_result = is_this_a_chart("images/chart.png", llm)
print(f"Is this a chart? {chart_result}")

print("done!")
