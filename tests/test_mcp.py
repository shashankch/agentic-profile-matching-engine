import unittest
import sys
from pathlib import Path

# Add project root directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentic_profile_matching import config
from agentic_profile_matching import fs_client
from agentic_profile_matching.mcp_client import mcp_client
from agentic_profile_matching.matching_agent import (
    extract_requirements_node,
    AgentState,
)


class TestMCPIntegration(unittest.TestCase):
    def setUp(self):
        # Cache original config flag
        self.original_use_mcp = config.USE_MCP

        # We will test in the data directory
        self.test_dir = Path(config.DATA_DIR) / "test_mcp_sandbox"
        self.test_dir.mkdir(exist_ok=True, parents=True)
        self.test_file = self.test_dir / "mcp_dummy.txt"
        self.test_file.write_text(
            "Hello from MCP test framework. Python programming is awesome."
        )

    def tearDown(self):
        # Restore configuration
        config.USE_MCP = self.original_use_mcp

        # Stop MCP clients if running
        mcp_client.stop()

        # Clean up sandbox
        if self.test_file.exists():
            self.test_file.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()

    def test_local_mode_routing(self):
        """Verify fs_client routes calls locally when USE_MCP = False"""
        config.USE_MCP = False

        result = fs_client.read_file(str(self.test_file.resolve()))
        self.assertTrue(result["success"])
        self.assertIn("Hello from MCP test framework", result["content"])
        self.assertEqual(result["metadata"]["filename"], "mcp_dummy.txt")

    def test_mcp_mode_routing_and_server(self):
        """Verify fs_client starts server and calls tools via MCP when USE_MCP = True"""
        config.USE_MCP = True

        # Direct read_file routing to MCP
        result = fs_client.read_file(str(self.test_file.resolve()))
        self.assertTrue(result["success"])
        self.assertIn("Hello from MCP test framework", result["content"])
        self.assertEqual(result["metadata"]["filename"], "mcp_dummy.txt")

        # Expose search_in_file
        search_res = fs_client.search_in_file(
            filepath=str(self.test_file.resolve()), keyword="Python", context_size=10
        )
        self.assertTrue(search_res["success"])
        self.assertEqual(search_res["keyword"], "Python")
        self.assertGreaterEqual(search_res["total_matches"], 1)

    def test_mcp_batch_processing(self):
        """Verify batch processing works concurrently via MCP"""
        config.USE_MCP = True

        file1 = self.test_dir / "batch1.txt"
        file2 = self.test_dir / "batch2.txt"
        file1.write_text("Content 1")
        file2.write_text("Content 2")

        try:
            paths = [str(file1.resolve()), str(file2.resolve())]
            batch_results = fs_client.batch_process(paths)

            self.assertEqual(len(batch_results), 2)
            self.assertTrue(batch_results[0]["success"])
            self.assertTrue(batch_results[1]["success"])
            self.assertEqual(batch_results[0]["content"], "Content 1")
            self.assertEqual(batch_results[1]["content"], "Content 2")
        finally:
            if file1.exists():
                file1.unlink()
            if file2.exists():
                file2.unlink()

    def test_multi_mcp_search_bonus(self):
        """Verify secondary search MCP server works through the Client Manager"""
        config.USE_MCP = True

        # Test mock search server web search tool
        search_res = mcp_client.call_tool(
            "search", "search_web", {"query": "John Doe Github"}
        )
        self.assertTrue(search_res["success"])
        self.assertEqual(search_res["query"], "John Doe Github")
        self.assertGreater(len(search_res["results"]), 0)

        # Test HR candidate notes tool
        notes_res = mcp_client.call_tool(
            "search", "fetch_candidate_notes", {"candidate_name": "Jane Smith"}
        )
        self.assertTrue(notes_res["success"])
        self.assertIn("Jane Smith", notes_res["candidate_name"])
        self.assertIn("Frontend", notes_res["notes"])

        # Test ChromaDB search tool
        db_res = mcp_client.call_tool(
            "search", "search_chroma_db", {"query": "Python", "limit": 2}
        )
        self.assertIn("success", db_res)

    def test_agent_error_fallback(self):
        """Verify that agent nodes record errors and recover using safe fallbacks when exceptions occur"""
        # Create state with bad credentials to guarantee LLM connection failure
        from langchain_core.messages import HumanMessage

        state = AgentState(
            messages=[
                HumanMessage(content="Analyze this job: Python Developer needed.")
            ],
            requirements={},
            shortlist=[],
            coarse_screen_limit=5,
            deep_screen_limit=3,
            recommendation_limit=2,
            current_round=1,
            errors=[],
            llm_provider="InvalidProvider",
            llm_model="gpt-4o",
            api_key="bad-invalid-key",
        )

        # Run node that executes LLM call
        result = extract_requirements_node(state)

        # Verify it handled exception, didn't crash, added error, and used fallback requirements
        self.assertIn("errors", result)
        self.assertGreater(len(result["errors"]), 0)
        self.assertTrue(
            any("Requirements extraction failed" in err for err in result["errors"])
        )
        self.assertEqual(result["requirements"]["title"], "Software Engineer")


if __name__ == "__main__":
    unittest.main()
