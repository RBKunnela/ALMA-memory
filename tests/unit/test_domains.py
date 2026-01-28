"""
Tests for the Domain Memory Factory module.
"""

from alma.domains import (
    DomainMemoryFactory,
    DomainSchema,
    EntityType,
    RelationshipType,
    get_coding_schema,
    get_general_schema,
    get_research_schema,
    get_sales_schema,
)


class TestEntityType:
    """Tests for EntityType dataclass."""

    def test_basic_creation(self):
        """Test basic entity type creation."""
        entity = EntityType(
            name="feature",
            description="A software feature",
            attributes=["status", "priority"],
        )

        assert entity.name == "feature"
        assert "status" in entity.attributes

    def test_validate_entity(self):
        """Test entity validation."""
        entity = EntityType(
            name="feature",
            description="A software feature",
            attributes=["status", "priority"],
            required_attributes=["status"],
        )

        # Valid entity
        errors = entity.validate_entity({"status": "done", "priority": 1})
        assert len(errors) == 0

        # Missing required attribute
        errors = entity.validate_entity({"priority": 1})
        assert len(errors) == 1
        assert "status" in errors[0]


class TestRelationshipType:
    """Tests for RelationshipType dataclass."""

    def test_basic_creation(self):
        """Test basic relationship creation."""
        rel = RelationshipType(
            name="tests",
            description="Test covers feature",
            source_type="test",
            target_type="feature",
        )

        assert rel.name == "tests"
        assert rel.source_type == "test"
        assert rel.target_type == "feature"


class TestDomainSchema:
    """Tests for DomainSchema dataclass."""

    def test_create_basic(self):
        """Test basic schema creation."""
        schema = DomainSchema.create(
            name="test_domain",
            description="A test domain",
            learning_categories=["cat1", "cat2"],
        )

        assert schema.id is not None
        assert schema.name == "test_domain"
        assert len(schema.learning_categories) == 2

    def test_add_entity_type(self):
        """Test adding entity types."""
        schema = DomainSchema.create("test", "Test domain")

        schema.add_entity_type(
            name="task",
            description="A task",
            attributes=["status", "priority"],
        )

        assert len(schema.entity_types) == 1
        assert schema.entity_types[0].name == "task"

    def test_add_relationship_type(self):
        """Test adding relationship types."""
        schema = DomainSchema.create("test", "Test domain")
        schema.add_entity_type("task", "A task")
        schema.add_entity_type("goal", "A goal")

        schema.add_relationship_type(
            name="achieves",
            description="Task achieves goal",
            source_type="task",
            target_type="goal",
        )

        assert len(schema.relationship_types) == 1
        assert schema.relationship_types[0].name == "achieves"

    def test_validate_valid_schema(self):
        """Test validation of valid schema."""
        schema = DomainSchema.create("test", "Test domain")
        schema.add_entity_type("task", "A task")
        schema.add_entity_type("goal", "A goal")
        schema.add_relationship_type("achieves", "achieves", "task", "goal")

        errors = schema.validate()
        assert len(errors) == 0

    def test_validate_invalid_relationship(self):
        """Test validation catches invalid relationships."""
        schema = DomainSchema.create("test", "Test domain")
        schema.add_entity_type("task", "A task")
        # Add relationship with non-existent target
        schema.add_relationship_type("achieves", "achieves", "task", "nonexistent")

        errors = schema.validate()
        assert len(errors) == 1
        assert "nonexistent" in errors[0]

    def test_is_category_allowed(self):
        """Test category allowance checking."""
        schema = DomainSchema.create(
            "test",
            "Test domain",
            learning_categories=["cat1", "cat2"],
            excluded_categories=["cat3"],
        )

        assert schema.is_category_allowed("cat1") is True
        assert schema.is_category_allowed("cat2") is True
        assert schema.is_category_allowed("cat3") is False
        assert schema.is_category_allowed("unknown") is False

    def test_to_dict_and_from_dict(self):
        """Test serialization roundtrip."""
        original = DomainSchema.create(
            "test",
            "Test domain",
            learning_categories=["cat1"],
        )
        original.add_entity_type("task", "A task", ["status"])
        original.add_relationship_type("relates", "relates", "task", "task")

        # Serialize
        data = original.to_dict()

        # Deserialize
        restored = DomainSchema.from_dict(data)

        assert restored.name == original.name
        assert restored.description == original.description
        assert len(restored.entity_types) == 1
        assert len(restored.relationship_types) == 1
        assert restored.learning_categories == original.learning_categories


class TestPrebuiltSchemas:
    """Tests for pre-built schemas."""

    def test_coding_schema(self):
        """Test coding schema structure."""
        schema = get_coding_schema()

        assert schema.name == "coding"
        assert len(schema.entity_types) >= 4
        assert len(schema.learning_categories) >= 5

        # Should have expected entities
        entity_names = [e.name for e in schema.entity_types]
        assert "feature" in entity_names
        assert "bug" in entity_names
        assert "test" in entity_names

        # Should validate
        errors = schema.validate()
        assert len(errors) == 0

    def test_research_schema(self):
        """Test research schema structure."""
        schema = get_research_schema()

        assert schema.name == "research"

        entity_names = [e.name for e in schema.entity_types]
        assert "paper" in entity_names
        assert "hypothesis" in entity_names
        assert "experiment" in entity_names

        errors = schema.validate()
        assert len(errors) == 0

    def test_sales_schema(self):
        """Test sales schema structure."""
        schema = get_sales_schema()

        assert schema.name == "sales"

        entity_names = [e.name for e in schema.entity_types]
        assert "lead" in entity_names
        assert "objection" in entity_names
        assert "deal" in entity_names

        errors = schema.validate()
        assert len(errors) == 0

    def test_general_schema(self):
        """Test general schema structure."""
        schema = get_general_schema()

        assert schema.name == "general"

        entity_names = [e.name for e in schema.entity_types]
        assert "task" in entity_names
        assert "goal" in entity_names

        errors = schema.validate()
        assert len(errors) == 0


class TestDomainMemoryFactory:
    """Tests for DomainMemoryFactory."""

    def test_list_schemas(self):
        """Test listing available schemas."""
        schemas = DomainMemoryFactory.list_schemas()

        assert "coding" in schemas
        assert "research" in schemas
        assert "sales" in schemas
        assert "general" in schemas

    def test_get_builtin_schema(self):
        """Test getting built-in schema."""
        schema = DomainMemoryFactory.get_schema("coding")

        assert schema is not None
        assert schema.name == "coding"

    def test_get_unknown_schema(self):
        """Test getting non-existent schema."""
        schema = DomainMemoryFactory.get_schema("nonexistent")
        assert schema is None

    def test_create_custom_schema(self):
        """Test creating custom schema."""
        schema = DomainMemoryFactory.create_schema(
            name="custom_domain",
            config={
                "description": "My custom domain",
                "entity_types": [
                    {
                        "name": "item",
                        "description": "An item",
                        "attributes": ["status"],
                    },
                ],
                "learning_categories": ["pattern1", "pattern2"],
            },
            register=True,
        )

        assert schema.name == "custom_domain"
        assert len(schema.entity_types) == 1

        # Should be retrievable
        retrieved = DomainMemoryFactory.get_schema("custom_domain")
        assert retrieved is not None

        # Cleanup
        DomainMemoryFactory.unregister_schema("custom_domain")

    def test_register_and_unregister(self):
        """Test schema registration lifecycle."""
        schema = DomainSchema.create("temp", "Temporary")
        DomainMemoryFactory.register_schema(schema)

        assert DomainMemoryFactory.get_schema("temp") is not None

        removed = DomainMemoryFactory.unregister_schema("temp")
        assert removed is True

        assert DomainMemoryFactory.get_schema("temp") is None

    def test_create_alma_instance(self):
        """Test creating ALMA instance from schema."""
        factory = DomainMemoryFactory()
        schema = get_general_schema()

        alma = factory.create_alma(
            schema=schema,
            project_id="test-project",
            embedding_provider="mock",
        )

        assert alma is not None
        assert alma._domain_schema == schema

    def test_create_alma_for_agent(self):
        """Test creating ALMA for specific agent."""
        factory = DomainMemoryFactory()

        alma = factory.create_alma_for_agent(
            schema_name="coding",
            agent="Helena",
            project_id="test-project",
            scope_restrictions=["backend_patterns"],
        )

        assert alma is not None
        # Should have scope restriction applied
        assert alma._domain_schema.is_category_allowed("backend_patterns") is False

    def test_convenience_functions(self):
        """Test convenience factory functions."""
        from alma.domains.factory import (
            create_coding_alma,
            create_general_alma,
            create_research_alma,
        )

        coding = create_coding_alma("proj", agent="dev", embedding_provider="mock")
        assert coding is not None
        assert coding._domain_schema.name == "coding"

        research = create_research_alma(
            "proj", agent="researcher", embedding_provider="mock"
        )
        assert research is not None
        assert research._domain_schema.name == "research"

        general = create_general_alma(
            "proj", agent="assistant", embedding_provider="mock"
        )
        assert general is not None
        assert general._domain_schema.name == "general"
