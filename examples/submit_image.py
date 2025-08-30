from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda

from motleycrew.common import LLMFramework
from motleycrew.common.llms import init_llm
from motleycrew.tools.image import image_to_human_message


def create_multimodal_messages(context: dict) -> list[HumanMessage]:
    """Create messages from context containing text_prompt and image_path.

    Args:
        context: Dict with 'text_prompt' and 'image_path' keys

    Returns:
        List of HumanMessages for text and image
    """
    messages = []

    # Add text message if provided
    if "text_prompt" in context:
        messages.append(HumanMessage(content=context["text_prompt"]))

    # Add image message if provided
    if "image_path" in context:
        messages.append(image_to_human_message(context["image_path"]))

    return messages


# Create a Runnable chain
multimodal_chain = RunnableLambda(create_multimodal_messages)

# Initialize the LLM
llm = init_llm(llm_framework=LLMFramework.LANGCHAIN)

# Create the full chain with LCEL syntax
chain = multimodal_chain | llm

# Define context (similar to your original approach)
context = {
    "text_prompt": "Describe the image",
    "image_path": "/home/james/Dropbox/Code/motleycrew/examples/images/girl.png",
}

# Invoke with LCEL syntax (like your original)
content = chain.invoke(context).content
print(content)
print("yay!")
