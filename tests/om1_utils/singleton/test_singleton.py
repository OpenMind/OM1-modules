import threading
from concurrent.futures import ThreadPoolExecutor

from om1_utils.singleton import singleton


class TestSingleton:
    """Tests for the singleton decorator."""

    def test_returns_same_instance(self):
        """Test that decorated class always returns the same instance."""

        @singleton
        class MyClass:
            pass

        a = MyClass()
        b = MyClass()
        assert a is b

    def test_preserves_init_args(self):
        """Test that the first call's arguments are used for the instance."""

        @singleton
        class MyClass:
            def __init__(self, value):
                self.value = value

        a = MyClass(42)
        b = MyClass(99)
        assert a.value == 42
        assert b.value == 42
        assert a is b

    def test_different_classes_have_different_instances(self):
        """Test that different decorated classes have separate instances."""

        @singleton
        class ClassA:
            pass

        @singleton
        class ClassB:
            pass

        a = ClassA()
        b = ClassB()
        assert a is not b

    def test_thread_safety(self):
        """Test that singleton is thread-safe under concurrent access."""
        call_count = 0

        @singleton
        class Counter:
            def __init__(self):
                nonlocal call_count
                call_count += 1
                self.id = call_count

        instances = []

        def create_instance():
            inst = Counter()
            instances.append(inst)

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_instance) for _ in range(50)]
            for f in futures:
                f.result()

        assert all(inst is instances[0] for inst in instances)
        assert call_count == 1

    def test_with_kwargs(self):
        """Test singleton with keyword arguments."""

        @singleton
        class Config:
            def __init__(self, host="localhost", port=8080):
                self.host = host
                self.port = port

        c = Config(host="0.0.0.0", port=9090)
        assert c.host == "0.0.0.0"
        assert c.port == 9090

    def test_instance_attributes_persist(self):
        """Test that attributes set on the instance persist across calls."""

        @singleton
        class Store:
            def __init__(self):
                self.data = []

        s1 = Store()
        s1.data.append("item1")

        s2 = Store()
        assert "item1" in s2.data
