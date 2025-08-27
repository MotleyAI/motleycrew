# """
# Tests for the SQL validation module.
# """
#
# import pytest
#
# from motleycrew.tools.sql_validation import (
#     SchemaValidator,
#     SQLValidator,
#     ValidationLevel,
#     ValidationError,
# )
#
#
# @pytest.fixture
# def schema_sql():
#     """Sample schema SQL for testing."""
#     return """
#     CREATE TABLE customers (
#         id INT PRIMARY KEY,
#         name VARCHAR(100) NOT NULL,
#         email VARCHAR(255) UNIQUE,
#         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#     );
#
#     CREATE TABLE orders (
#         id INT PRIMARY KEY,
#         customer_id INT REFERENCES customers(id),
#         amount DECIMAL(10,2) NOT NULL,
#         status VARCHAR(20) DEFAULT 'pending',
#         created_at TIMESTAMP
#     );
#     """
#
#
# @pytest.fixture
# def schema_dict():
#     """Sample schema dictionary for testing."""
#     return {
#         "customers": {
#             "id": "INT",
#             "name": "VARCHAR(100)",
#             "email": "VARCHAR(255)",
#             "created_at": "TIMESTAMP",
#         },
#         "orders": {
#             "id": "INT",
#             "customer_id": "INT",
#             "amount": "DECIMAL(10,2)",
#             "status": "VARCHAR(20)",
#             "created_at": "TIMESTAMP",
#         },
#     }
#
#
# @pytest.fixture
# def valid_query():
#     """A valid SQL query for testing."""
#     return """
#     SELECT c.name, SUM(o.amount) as total_spent
#     FROM customers c
#     JOIN orders o ON c.id = o.customer_id
#     WHERE o.status = 'completed'
#     GROUP BY c.name
#     """
#
#
# @pytest.fixture
# def invalid_query_syntax():
#     """A SQL query with syntax errors for testing."""
#     return """
#     SELECT c.name, SUM(o.amount as total_spent
#     FROM customers c
#     JOIN orders o ON c.id = o.customer_id
#     WHERE o.status = 'completed'
#     GROUP BY c.name
#     """
#
#
# @pytest.fixture
# def invalid_query_schema():
#     """A SQL query with schema errors for testing."""
#     return """
#     SELECT c.name, c.phone_number, SUM(o.amount) as total_spent
#     FROM customers c
#     JOIN orders o ON c.id = o.customer_id
#     WHERE o.status = 'completed'
#     GROUP BY c.name, c.phone_number
#     """
#
#
# @pytest.fixture
# def invalid_query_table():
#     """A SQL query with non-existent table for testing."""
#     return """
#     SELECT c.name, SUM(o.amount) as total_spent
#     FROM customers c
#     JOIN invoices i ON c.id = i.customer_id
#     WHERE o.status = 'completed'
#     GROUP BY c.name
#     """
#
#
# class TestSchemaValidator:
#     """Tests for the SchemaValidator class."""
#
#     def test_load_schema_from_sql(self, schema_sql):
#         """Test loading schema from SQL."""
#         validator = SchemaValidator()
#         schema_dict = validator.load_schema_from_sql(schema_sql)
#
#         assert "customers" in schema_dict
#         assert "orders" in schema_dict
#         assert "id" in schema_dict["customers"]
#         assert "name" in schema_dict["customers"]
#         assert "email" in schema_dict["customers"]
#         assert "created_at" in schema_dict["customers"]
#         assert "id" in schema_dict["orders"]
#         assert "customer_id" in schema_dict["orders"]
#         assert "amount" in schema_dict["orders"]
#         assert "status" in schema_dict["orders"]
#         assert "created_at" in schema_dict["orders"]
#
#     def test_load_schema_from_dict(self, schema_dict):
#         """Test loading schema from dictionary."""
#         validator = SchemaValidator()
#         validator.load_schema_from_dict(schema_dict)
#
#         assert validator.schema == schema_dict
#
#     def test_validate_query_valid(self, schema_sql, valid_query):
#         """Test validating a valid query."""
#         validator = SchemaValidator()
#         validator.load_schema_from_sql(schema_sql)
#
#         is_valid, errors, optimized_query = validator.validate_query(valid_query)
#
#         assert is_valid is True
#         assert len(errors) == 0
#         assert optimized_query is not None
#
#     def test_validate_query_syntax_error(self, schema_sql, invalid_query_syntax):
#         """Test validating a query with syntax errors."""
#         validator = SchemaValidator()
#         validator.load_schema_from_sql(schema_sql)
#
#         is_valid, errors, optimized_query = validator.validate_query(invalid_query_syntax)
#
#         assert is_valid is False
#         assert len(errors) > 0
#         assert "Syntax Error" in errors[0].error_type
#         assert optimized_query is None
#
#     def test_validate_query_schema_error(self, schema_sql, invalid_query_schema):
#         """Test validating a query with schema errors."""
#         validator = SchemaValidator()
#         validator.load_schema_from_sql(schema_sql)
#
#         is_valid, errors, optimized_query = validator.validate_query(invalid_query_schema)
#
#         assert is_valid is False
#         assert len(errors) > 0
#         assert optimized_query is None
#
#     def test_validate_query_table_error(self, schema_sql, invalid_query_table):
#         """Test validating a query with non-existent table."""
#         validator = SchemaValidator()
#         validator.load_schema_from_sql(schema_sql)
#
#         is_valid, errors, optimized_query = validator.validate_query(invalid_query_table)
#
#         assert is_valid is False
#         assert len(errors) > 0
#         assert optimized_query is None
#
#     def test_get_detailed_validation_errors(self, schema_sql, invalid_query_schema):
#         """Test getting detailed validation errors."""
#         validator = SchemaValidator()
#         validator.load_schema_from_sql(schema_sql)
#
#         errors = validator.get_detailed_validation_errors(invalid_query_schema)
#
#         assert len(errors) > 0
#         assert any("phone_number" in error.message for error in errors)
#
#
# class TestSQLValidator:
#     """Tests for the SQLValidator class."""
#
#     def test_validate_syntax(self, valid_query, invalid_query_syntax):
#         """Test validating syntax."""
#         is_valid, error = SQLValidator.validate_syntax(valid_query)
#         assert is_valid is True
#         assert error is None
#
#         is_valid, error = SQLValidator.validate_syntax(invalid_query_syntax)
#         assert is_valid is False
#         assert error is not None
#
#     def test_validate_query_against_schema(self, schema_sql, valid_query, invalid_query_schema):
#         """Test validating a query against a schema."""
#         is_valid, errors, _ = SQLValidator.validate_query_against_schema(valid_query, schema_sql)
#         assert is_valid is True
#         assert len(errors) == 0
#
#         is_valid, errors, _ = SQLValidator.validate_query_against_schema(invalid_query_schema, schema_sql)
#         assert is_valid is False
#         assert len(errors) > 0
#
#     def test_validate_query_against_schema_dict(self, schema_dict, valid_query, invalid_query_schema):
#         """Test validating a query against a schema dictionary."""
#         is_valid, errors, _ = SQLValidator.validate_query_against_schema_dict(valid_query, schema_dict)
#         assert is_valid is True
#         assert len(errors) == 0
#
#         is_valid, errors, _ = SQLValidator.validate_query_against_schema_dict(invalid_query_schema, schema_dict)
#         assert is_valid is False
#         assert len(errors) > 0
#
#     def test_validation_levels(self, schema_sql, valid_query):
#         """Test different validation levels."""
#         # Syntax only
#         is_valid, errors, _ = SQLValidator.validate_query_against_schema(
#             valid_query, schema_sql, level=ValidationLevel.SYNTAX_ONLY
#         )
#         assert is_valid is True
#         assert len(errors) == 0
#
#         # Schema only
#         is_valid, errors, _ = SQLValidator.validate_query_against_schema(
#             valid_query, schema_sql, level=ValidationLevel.SCHEMA_ONLY
#         )
#         assert is_valid is True
#         assert len(errors) == 0
#
#         # Full validation
#         is_valid, errors, _ = SQLValidator.validate_query_against_schema(
#             valid_query, schema_sql, level=ValidationLevel.FULL
#         )
#         assert is_valid is True
#         assert len(errors) == 0
#
#     def test_format_query(self, valid_query):
#         """Test formatting a query."""
#         formatted_query = SQLValidator.format_query(valid_query)
#         assert formatted_query is not None
#         assert "SELECT" in formatted_query
#         assert "FROM" in formatted_query
#         assert "JOIN" in formatted_query
#         assert "WHERE" in formatted_query
#         assert "GROUP BY" in formatted_query
#
#     def test_transpile_query(self, valid_query):
#         """Test transpiling a query between dialects."""
#         transpiled_query = SQLValidator.transpile_query(valid_query, "postgres", "mysql")
#         assert transpiled_query is not None
#         assert "SELECT" in transpiled_query
#         assert "FROM" in transpiled_query
#         assert "JOIN" in transpiled_query
#         assert "WHERE" in transpiled_query
#         assert "GROUP BY" in transpiled_query
#
#
# class TestValidationError:
#     """Tests for the ValidationError class."""
#
#     def test_validation_error_str(self):
#         """Test string representation of ValidationError."""
#         error = ValidationError(
#             message="Unknown column 'phone_number'",
#             error_type="Schema Validation Error",
#             line=3,
#             column=15,
#             context="SELECT c.name, c.phone_number"
#         )
#
#         error_str = str(error)
#         assert "Schema Validation Error" in error_str
#         assert "Unknown column 'phone_number'" in error_str
#         assert "line 3" in error_str
#         assert "column 15" in error_str
#         assert "SELECT c.name, c.phone_number" in error_str
