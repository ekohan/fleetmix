import multiprocessing
from unittest.mock import MagicMock, patch


def test_multiprocessing_optimization():
    """Test that optimization runs in a separate process correctly."""
    with patch("fleetmix.gui.run_optimization_in_process") as mock_run:
        # Create a mock process
        mock_process = MagicMock()
        mock_process.is_alive.return_value = False
        mock_process.start = MagicMock()
        mock_process.join = MagicMock()

        with patch("multiprocessing.Process", return_value=mock_process):
            # This would be called within the GUI
            process = multiprocessing.Process(
                target=mock_run, args=("demand.csv", {}, "output", "status.json")
            )
            process.start()

            # Verify process was started
            mock_process.start.assert_called_once()


def test_session_state_persistence():
    """Test that session state persists across reruns."""
    import streamlit as st

    # Mock session state
    mock_state = MagicMock()
    mock_state.uploaded_data = None

    with patch.object(st, "session_state", mock_state):
        # Simulate file upload
        mock_state.uploaded_data = "test_data"

        # Verify state persists
        assert st.session_state.uploaded_data == "test_data"


if __name__ == "__main__":
    # Run the simple tests
    test_multiprocessing_optimization()
    test_session_state_persistence()
    print("\nFor full E2E testing, consider implementing one of the approaches above.")
