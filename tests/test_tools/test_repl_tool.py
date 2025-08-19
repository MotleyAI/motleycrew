from motleycrew.tools.code import PythonREPLTool


class TestREPLTool:
    def test_repl_tool(self):
        repl_tool = PythonREPLTool()
        repl_tool_input_fields = list(repl_tool.tool.args_schema.model_fields.keys())

        assert repl_tool_input_fields == ["command"]
        assert repl_tool.invoke({repl_tool_input_fields[0]: "print(1)"}).strip() == "1"

    def test_repl_tool_multiple_calls(self):
        repl_tool = PythonREPLTool()
        assert repl_tool.invoke({"command": "print(a := 1)"}).strip() == "1"
        assert repl_tool.invoke({"command": "print(a)"}).strip() == "1"

    def test_repl_tool_multiline_input_with_import(self):
        repl_tool = PythonREPLTool()
        command = """import math
print(math.sqrt(16))
"""
        assert repl_tool.invoke({"command": command}).strip() == "4.0"
