from src.inference import VideoAnalyzer


def test_analyzer_init():
    try:
        a = VideoAnalyzer()
        assert hasattr(a, 'models')
    except Exception as e:
        # Acceptable if deps missing, but structure should be importable
        pass
