import unittest
from .federated_helpers import get_all_federated_loaders
from typing import List

LOADERS = get_all_federated_loaders()


def client_fn(cid: int):
    return cid % len(LOADERS)


class ClientsLoadersTest(unittest.TestCase):
    def __init__(
        self,
        methodName: str,
        correct_order: List[str] = [],
        wrong_order: List[str] = [],
    ) -> None:
        super().__init__(methodName)
        self.correct_order = correct_order
        self.wrong_order = wrong_order

    def test_get_clients_names_correct_order(self):
        for cid, name in enumerate(self.correct_order):
            idx = client_fn(cid)
            (_, client_name), (_, _) = LOADERS[idx]
            self.assertEqual(client_name, name)

    def test_get_clients_names_wrong_order(self):
        for cid, name in enumerate(self.wrong_order):
            idx = client_fn(cid)
            (_, client_name), (_, _) = LOADERS[idx]
            self.assertNotEqual(client_name, name)


if __name__ == "__main__":
    correct_order = ["client-1: ToN", "client-2: BoT", "client-3: UNSW"]
    wrong_order = ["client-2: BoT", "client-3: UNSW", "client-1: ToN"]

    suite = unittest.TestSuite()
    suite.addTests(
        [
            ClientsLoadersTest(
                "test_get_clients_names_correct_order", correct_order=correct_order
            ),
            ClientsLoadersTest(
                "test_get_clients_names_wrong_order", wrong_order=wrong_order
            ),
        ]
    )
    runner = unittest.TextTestRunner()
    runner.run(suite)
