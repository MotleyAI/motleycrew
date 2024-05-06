"""
Code derived from: https://github.com/run-llama/llama_index/blob/802064aee72b03ab38ead0cda780cfa3e37ce728/llama-index-integrations/graph_stores/llama-index-graph-stores-kuzu/llama_index/graph_stores/kuzu/base.py
Kùzu graph store index.
"""

import logging
from time import sleep

from typing import Any, Dict, List, Optional, Type, TypeVar

import kuzu
from kuzu import PreparedStatement, QueryResult
import json

from motleycrew.storage import MotleyGraphStore
from motleycrew.storage import MotleyGraphNode


class MotleyKuzuGraphStore(MotleyGraphStore):
    ID_ATTR = "_id"

    JSON_CONTENT_PREFIX = "JSON__"

    PYTHON_TO_CYPHER_TYPES_MAPPING = {
        int: "INT64",  # TODO: enforce size when creating and updating nodes and relations
        Optional[int]: "INT64",
        str: "STRING",
        Optional[str]: "STRING",
        float: "DOUBLE",
        Optional[float]: "DOUBLE",
        bool: "BOOLEAN",
        Optional[bool]: "BOOLEAN",
    }

    def __init__(self, database: Any) -> None:
        self.database = database
        self.connection = kuzu.Connection(database)

    def _execute_query(
        self, query: str | PreparedStatement, parameters: Optional[dict[str, Any]] = None
    ) -> QueryResult:
        """
        Execute a query, logging it for debugging purposes
        """
        logging.debug("Executing query: %s", query)
        if parameters:
            logging.debug("with parameters: %s", parameters)

        # TODO: retries?
        return self.connection.execute(query=query, parameters=parameters)

    def _check_node_table_exists(self, label: str):
        return label in self.connection._get_node_table_names()

    def _check_rel_table_exists(
        self, from_label: str, to_label: str, rel_label: Optional[str] = None
    ):
        for row in self.connection._get_rel_table_names():
            if (
                (rel_label is None or row["name"] == rel_label)
                and row["src"] == from_label
                and row["dst"] == to_label
            ):
                return True
        return False

    def _get_node_property_names(self, label: str):
        return self.connection._get_node_property_names(table_name=label)

    def _ensure_node_table(self, node: MotleyGraphNode) -> str:
        """
        Create a table for storing nodes of that class if such does not already exist.
        If it does exist, create all missing columns.
        Return the table name.
        """
        table_name = node.get_label()
        if not self._check_node_table_exists(table_name):
            logging.info("Node table %s does not exist in the database, creating", table_name)
            self._execute_query(
                "CREATE NODE TABLE {} (id SERIAL, PRIMARY KEY(id))".format(table_name)
            )

        # Create missing property columns
        existing_property_names = self._get_node_property_names(node.get_label())
        for field_name, field in node.model_fields.items():
            if field_name not in existing_property_names:
                logging.info(
                    "Property %s not present in table for label %s, creating",
                    field_name,
                    node.get_label(),
                )
                cypher_type, is_json = (
                    MotleyKuzuGraphStore._get_cypher_type_and_is_json_by_python_type_annotation(
                        field.annotation
                    )
                )

                self._execute_query(
                    "ALTER TABLE {} ADD {} {}".format(table_name, field_name, cypher_type)
                )
        return table_name

    def _ensure_relation_table(
        self, from_node: MotleyGraphNode, to_node: MotleyGraphNode, label: str
    ):
        """
        Create a table for storing relations from from_node-like nodes to to_node-like nodes,
        if such does not already exist.
        """
        if not self._check_rel_table_exists(
            from_label=from_node.get_label(), to_label=to_node.get_label(), rel_label=label
        ):
            logging.info(
                "Relation table %s from %s to %s does not exist in the database, creating",
                label,
                from_node.get_label(),
                to_node.get_label(),
            )

            self._execute_query(
                "CREATE REL TABLE {} (FROM {} TO {})".format(
                    label, from_node.get_label(), to_node.get_label()
                )
            )

    def check_node_exists_by_class_and_id(
        self, node_class: Type[MotleyGraphNode], node_id: int
    ) -> bool:
        """
        Check if a node of given class with given id is present in the database.
        """
        if not self._check_node_table_exists(node_class.get_label()):
            return False

        is_exists_result = self._execute_query(
            "MATCH (n:{}) WHERE n.id = $node_id RETURN n.id".format(node_class.get_label()),
            {"node_id": node_id},
        )
        return is_exists_result.has_next()

    def check_node_exists(self, node: MotleyGraphNode) -> bool:
        """
        Check if the given node is present in the database.
        """
        if node.id is None:
            return False  # for cases when id attribute is not set => node does not exist

        return self.check_node_exists_by_class_and_id(node_class=node.__class__, node_id=node.id)

    def check_relation_exists(
        self, from_node: MotleyGraphNode, to_node: MotleyGraphNode, label: Optional[str] = None
    ) -> bool:
        """
        Check if a relation exists between two nodes with given label.
        """
        if from_node.id is None or to_node.id is None:
            return False

        if (
            not self._check_node_table_exists(from_node.get_label())
            or not self._check_node_table_exists(to_node.get_label())
            or not self._check_rel_table_exists(
                from_label=from_node.get_label(), to_label=to_node.get_label(), rel_label=label
            )
        ):
            return False

        query = (
            "MATCH (n1:{})-[r{}]->(n2:{}) "
            "WHERE n1.id = $from_node_id AND n2.id = $to_node_id "
            "RETURN r".format(
                from_node.get_label(),
                (":" + label) if label else "",
                to_node.get_label(),
            )
        )
        parameters = {
            "from_node_id": from_node.id,
            "to_node_id": to_node.id,
        }

        is_exists_result = self._execute_query(query=query, parameters=parameters)
        return is_exists_result.has_next()

    def get_node_by_class_and_id(
        self, node_class: Type[MotleyGraphNode], node_id: int
    ) -> Optional[MotleyGraphNode]:
        """
        Retrieve the node of given class with given id if it is present in the database.
        Otherwise, return None.
        """
        if not self._check_node_table_exists(node_class.get_label()):
            return None

        query = """
                    MATCH (n:{})
                    WHERE n.id = $node_id
                    RETURN n;
                """.format(
            node_class.get_label()
        )
        query_result = self._execute_query(query, {"node_id": node_id})

        if query_result.has_next():
            row = query_result.get_next()
            node_dict = row[0]
            for field_name, value in node_dict.copy().items():
                if isinstance(value, str) and value.startswith(
                    MotleyKuzuGraphStore.JSON_CONTENT_PREFIX
                ):
                    logging.debug(
                        "Value for field %s is marked as JSON, attempting to deserialize: %s",
                        field_name,
                        value,
                    )
                    node_dict[field_name] = json.loads(
                        value[len(MotleyKuzuGraphStore.JSON_CONTENT_PREFIX) :]
                    )

            if node_class is not None:
                return node_class.parse_obj(node_dict)
            return node_dict

    def insert_node(self, node: MotleyGraphNode) -> MotleyGraphNode:
        """
        Insert a new node, populate its id and freeze it.
        If node table or some columns do not exist, this method also creates them.
        """
        assert node.id is None, "Entity has its id set, looks like it is already in the DB"

        self._ensure_node_table(node)
        logging.info("Inserting new node with label %s: %s", node.get_label(), node)

        cypher_mapping, parameters = MotleyKuzuGraphStore._node_to_cypher_mapping_with_parameters(
            node
        )
        create_result = self._execute_query(
            "CREATE (n:{} {}) RETURN n".format(node.get_label(), cypher_mapping),
            parameters=parameters,
        )
        assert create_result.has_next()
        logging.info("Node created OK")

        created_object = create_result.get_next()[0]
        created_object_id = created_object.get("id")
        assert created_object_id is not None, "BUG: created object ID was not returned: {}".format(
            created_object
        )

        MotleyKuzuGraphStore._set_node_id(node=node, node_id=created_object_id)
        MotleyKuzuGraphStore._freeze_node(node)
        return node

    def create_relation(
        self, from_node: MotleyGraphNode, to_node: MotleyGraphNode, label: str
    ) -> None:
        """
        Create a relation between existing nodes.
        If relation table does not exist, this method also creates them.
        """
        assert self.check_node_exists(from_node), (
            "From-node is not present in the database, "
            "consider using upsert_triplet() for such cases"
        )
        assert self.check_node_exists(to_node), (
            "To-node is not present in the database, "
            "consider using upsert_triplet() for such cases"
        )

        self._ensure_relation_table(from_node=from_node, to_node=to_node, label=label)

        logging.info(
            "Creating relation %s from %s:%s to %s:%s",
            label,
            from_node.get_label(),
            from_node.id,
            to_node.get_label(),
            to_node.id,
        )

        create_result = self._execute_query(
            (
                "MATCH (n1:{}), (n2:{}) WHERE n1.id = $from_id AND n2.id = $to_id "
                "CREATE (n1)-[r:{}]->(n2) "
                "RETURN r"
            ).format(from_node.get_label(), to_node.get_label(), label),
            {
                "from_id": from_node.id,
                "to_id": to_node.id,
            },
        )
        assert create_result.has_next()
        logging.info("Relation created OK")

    def upsert_triplet(self, from_node: MotleyGraphNode, to_node: MotleyGraphNode, label: str):
        """
        Create a relation with a given label between nodes, if such does not already exist.
        If the nodes do not already exist, create them too.
        This method also creates and/or updates all necessary tables.
        """
        if not self.check_node_exists(from_node):
            logging.info("Node %s does not exist, creating", from_node)
            self.insert_node(from_node)

        if not self.check_node_exists(to_node):
            logging.info("Node %s does not exist, creating", to_node)
            self.insert_node(to_node)

        if not self.check_relation_exists(from_node=from_node, to_node=to_node, label=label):
            logging.info("Relation from %s to %s does not exist, creating", from_node, to_node)
            self.create_relation(from_node=from_node, to_node=to_node, label=label)

    def delete_node(self, node: MotleyGraphNode) -> None:
        """
        Delete a given node and its relations.
        """

        def inner_delete_relations(node_label: str, node_id: int) -> None:
            if not self.connection._get_rel_table_names():
                # Avoid Kuzu error when no relation tables exist in the database
                return

            # Undirected relation removal is not supported for some reason
            self._execute_query(
                "MATCH (n:{})-[r]->() WHERE n.id = $node_id DELETE r".format(node_label),
                {"node_id": node_id},
            )
            self._execute_query(
                "MATCH (n:{})<-[r]-() WHERE n.id = $node_id DELETE r".format(node_label),
                {"node_id": node_id},
            )

        def inner_delete_node(node_label: str, node_id: int) -> None:
            self._execute_query(
                "MATCH (n:{}) WHERE n.id = $node_id DELETE n".format(node_label),
                {"node_id": node_id},
            )

        assert self.check_node_exists(node), "Cannot delete nonexistent node: {}".format(node)

        inner_delete_relations(node_label=node.get_label(), node_id=node.id)
        inner_delete_node(node_label=node.get_label(), node_id=node.id)

        MotleyKuzuGraphStore._unfreeze_node(node)
        MotleyKuzuGraphStore._set_node_id(node, None)

    def set_property(
        self, node: MotleyGraphNode, property_name: str, property_value: Any
    ) -> MotleyGraphNode:
        """
        Set a property to a node. Also sets the property in the Python object.
        """
        if property_value is None:
            # TODO: remove after updating Kuzu to v0.3.3 (https://github.com/kuzudb/kuzu/pull/3098)
            raise Exception("Kuzu does not support NoneType parameters for properties for now")

        existing_property_names = self._get_node_property_names(node.get_label())

        assert property_name in node.model_fields, "No such field in node model {}: {}".format(
            node.__class__.__name__, property_name
        )

        assert self.check_node_exists(node)
        assert property_name in existing_property_names, "No such field in DB table {}: {}".format(
            node.get_label(), property_name
        )

        # Running Pydantic validation beforehand to avoid writing invalid values to the DB
        node.__pydantic_validator__.validate_assignment(
            node.model_construct(), property_name, property_value
        )

        _, is_json = MotleyKuzuGraphStore._get_cypher_type_and_is_json_by_python_type_annotation(
            node.model_fields[property_name].annotation
        )

        db_property_name = property_name
        if is_json:
            db_property_value = MotleyKuzuGraphStore.JSON_CONTENT_PREFIX + json.dumps(
                property_value
            )
        else:
            db_property_value = property_value

        query = """
                    MATCH (n:{})
                    WHERE n.id = $node_id
                    SET n.{} = $property_value RETURN n;
                """.format(
            node.get_label(), db_property_name
        )

        query_result = self._execute_query(
            query,
            {"node_id": node.id, "property_value": db_property_value},
        )
        assert query_result.has_next()
        row = query_result.get_next()
        node_dict = row[0]
        assert node_dict[db_property_name] == db_property_value

        # Now set the property value in the Python object
        MotleyKuzuGraphStore._unfreeze_node(node)
        setattr(node, property_name, property_value)
        MotleyKuzuGraphStore._freeze_node(node)
        return node

    def run_cypher_query(self, query: str, parameters: Optional[dict] = None) -> list[list]:
        """
        Run a Cypher query and return the results.
        """
        query_result = self._execute_query(query=query, parameters=parameters)
        retval = []
        while query_result.has_next():
            retval.append(query_result.get_next())
        return retval

    @staticmethod
    def _set_node_id(node: MotleyGraphNode, node_id: Optional[int]) -> None:
        setattr(node, MotleyKuzuGraphStore.ID_ATTR, node_id)

    @staticmethod
    def _freeze_node(node: MotleyGraphNode) -> None:
        """
        Make the node immutable by enabling its model_config["frozen"].
        Depends on the corresponding Pydantic feature.
        See https://docs.pydantic.dev/latest/concepts/models/#faux-immutability
        """
        assert (
            node.id is not None
        ), "Cannot freeze node because its id is not set, it may not be in the database yet"

        node.model_config["frozen"] = True

    @staticmethod
    def _unfreeze_node(node: MotleyGraphNode) -> None:
        """
        Reverse operation to _freeze_node().
        """
        node.model_config["frozen"] = False

    @staticmethod
    def _node_to_cypher_mapping_with_parameters(node: MotleyGraphNode) -> tuple[str, dict]:
        node_dict = node.model_dump()

        parameters = {}

        cypher_mapping = "{"
        for field_name, value in node_dict.items():
            if value is None:
                # TODO: remove after updating Kuzu to v0.3.3
                # (https://github.com/kuzudb/kuzu/pull/3098)
                continue

            _, is_json = (
                MotleyKuzuGraphStore._get_cypher_type_and_is_json_by_python_type_annotation(
                    node.model_fields[field_name].annotation
                )
            )
            if is_json and value is not None:
                value = json.dumps(value)
                value = MotleyKuzuGraphStore.JSON_CONTENT_PREFIX + value

            cypher_mapping += f"{field_name}: ${field_name}, "
            parameters[field_name] = value

        cypher_mapping = cypher_mapping.rstrip(", ") + "}"
        return cypher_mapping, parameters

    @staticmethod
    def _get_cypher_type_and_is_json_by_python_type_annotation(
        annotation: Type,
    ) -> tuple[str, bool]:
        """
        Determine suitable Cypher data type by Python/Pydantic type annotation,
        and whether the data should be stored in JSON-serialized strings.
        """
        cypher_type = MotleyKuzuGraphStore.PYTHON_TO_CYPHER_TYPES_MAPPING.get(annotation)
        if not cypher_type:
            logging.warning(
                "No known Cypher type matching annotation %s, will use JSON string",
                annotation,
            )
            return MotleyKuzuGraphStore.PYTHON_TO_CYPHER_TYPES_MAPPING[str], True
        return cypher_type, False

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str,
    ) -> "MotleyKuzuGraphStore":
        """Load from persist dir."""
        try:
            import kuzu
        except ImportError:
            raise ImportError("Please install kuzu: pip install kuzu")
        database = kuzu.Database(persist_dir)
        return cls(database)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "MotleyKuzuGraphStore":
        """Initialize graph store from configuration dictionary.

        Args:
            config_dict: Configuration dictionary.

        Returns:
            Graph store.
        """
        return cls(**config_dict)


if __name__ == "__main__":
    from pathlib import Path
    import shutil

    here = Path(__file__).parent
    db_path = here / "test1"
    shutil.rmtree(db_path, ignore_errors=True)
    db = kuzu.Database(str(db_path))
    graph_store = MotleyKuzuGraphStore(db)

    class Question(MotleyGraphNode):
        question: str
        answer: Optional[str] = None
        context: Optional[List[str]] = None

    q1 = Question(question="q1")
    graph_store.insert_node(q1)
    assert getattr(q1, "_id", None) is not None
    q1_id = q1._id

    assert graph_store.check_node_exists(q1)
    assert graph_store.check_node_exists_by_class_and_id(node_class=Question, node_id=q1_id)

    q2 = Question(question="q2", answer="a2")
    graph_store.upsert_triplet(from_node=q1, to_node=q2, label="p")
    assert getattr(q2, "_id", None) is not None
    q2_id = q2._id

    assert graph_store.check_relation_exists(from_node=q1, to_node=q2, label="p")
    assert not graph_store.check_relation_exists(from_node=q2, to_node=q1)

    graph_store.delete_node(q1)
    assert not graph_store.check_node_exists(q1)
    assert graph_store.get_node_by_class_and_id(node_class=Question, node_id=q1_id) is None

    graph_store.set_property(q2, property_name="context", property_value=["abc", "def"])
    assert q2.context == ["abc", "def"]

    assert graph_store.get_node_by_class_and_id(node_class=Question, node_id=q2_id).context == [
        "abc",
        "def",
    ]

    print(f"docker run -p 8000:8000  -v {db_path}:/database --rm kuzudb/explorer: latest")
    print("MATCH (A)-[r]->(B) RETURN *;")
