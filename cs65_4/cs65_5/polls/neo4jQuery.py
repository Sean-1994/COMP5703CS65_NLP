from neo4j import GraphDatabase
import logging
from neo4j.exceptions import ServiceUnavailable

uri = "neo4j+s://227816f4.databases.neo4j.io"
user = "neo4j"
password = "0A2TcqorfsDg-ai2Brr3YUDuHnYR3UGyeOm5dcWKHgo"

class App:
    label_relation = {"Symptom": "HAS_SYMPTON", "Cause": "CAUSE", "Disease": "COMPLICATION", "Prevention": "PREVENT",
                      "Treatment": "TREAT", "Diagnosis": "DIAGNOSE"}

    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_relationship(self, write_disease_name, write_node_label, write_node_name):
        with self.driver.session() as session:
            result = session.write_transaction(
                self._create_and_return_relationship, write_disease_name, write_node_label, write_node_name)
            for row in result:
                result_creation = (
                    "Created relation:{r} between: {p1}, {p2}".format(r=App.label_relation[write_node_label],
                                                                      p1=row['p1'], p2=row['p2']))
                print(result_creation)

            return result_creation

    @staticmethod
    def _create_and_return_relationship(tx, write_disease_name, write_node_label, write_node_name):
        write_relation = App.label_relation[write_node_label]
        query = (
            "MERGE (p1:Disease { name: $write_disease_name }) "
            "WITH p1 "
            "CALL apoc.merge.node([ $write_node_label ], {name: $write_node_name}) YIELD node as p2 "
            "CALL apoc.create.relationship(p1, $write_relation, {}, p2) YIELD rel "
            "RETURN p1, p2"
        )
        result = tx.run(query, write_disease_name=write_disease_name, write_node_label=write_node_label,
                        write_node_name=write_node_name, write_relation=write_relation)
        try:
            return [{"p1": row["p1"]["name"], "p2": row["p2"]["name"]}
                    for row in result]
        # Capture any errors along with the query and data for traceability
        except ServiceUnavailable as exception:
            logging.error("{query} raised an error: \n {exception}".format(
                query=query, exception=exception))
            raise

    def find_node(self, disease_name):
        with self.driver.session() as session:
            result = session.read_transaction(self._find_and_return_node, disease_name)
            for node in result:
                print("Relation: {l}, Node content: {n}".format(l=node[0], n=node[1]))
                result_find = ("Relation: {l}, Node content: {n}".format(l=node[0], n=node[1]))

            return result_find

    @staticmethod
    def _find_and_return_node(tx, disease_name):
        query = (
            "MATCH relation=(d:Disease)--(node)"
            "WHERE d.name = $disease_name "
            "RETURN relation"
        )
        result = tx.run(query, disease_name=disease_name)
        # ipdb.set_trace()
        return_list = []
        for row in result:
            return_dict = row.data()
            node_label = return_dict['relation'][1]
            node_value = return_dict['relation'][2]
            # print(node_label,node_value)
            return_list.append([node_label, node_value])
        return return_list

    def delete_node(self, node_label, node_name):
        with self.driver.session() as session:
            result = session.read_transaction(self._delete_node, node_label, node_name)
            # print(result)
            print("Node", node_name, "and connected relation has been deleted")
            result_delete = ("Node", node_name, "and connected relation has been deleted")

            return result_delete

    @staticmethod
    def _delete_node(tx, node_label, node_name):
        query = (
            "MATCH (n {name: $node_name })"
            "DETACH DELETE n"
        )
        result = tx.run(query, node_label=node_label, node_name=node_name)
        return result

    def modify_node(self, node_label, node_name, after_name):
        with self.driver.session() as session:
            result = session.read_transaction(self._modify_node, node_label, node_name, after_name)
            print("{n1} has been changed to {n2}".format(n1=node_name, n2=after_name))
            result_modify = ("{n1} has been changed to {n2}".format(n1=node_name, n2=after_name))

            return result_modify

    @staticmethod
    def _modify_node(tx, node_label, node_name, after_name):
        query = (
            "MATCH (n{name: $node_name })"
            "SET n.name = $after_name "
            "RETURN n"
        )
        result = tx.run(query, node_label=node_label, node_name=node_name, after_name=after_name)
        return result