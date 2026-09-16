import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingestion.loader import load_sales_file
from src.analytics.engine import run_analytics
from src.reporting.reports import generate_management_report
from src.ai.orchestrator import SalesAnalystAgent

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "sample" / "fmcg_sales_sample.csv"


def test_management_report_generates_without_claude():
    df = load_sales_file(SAMPLE)
    analytics = run_analytics(df)
    report = generate_management_report(analytics)
    assert "Executive Summary" in report
    assert "Total revenue" in report
    assert "Target Performance" in report


def test_orchestrator_answers_without_claude_when_disabled():
    df = load_sales_file(SAMPLE)
    agent = SalesAnalystAgent(df)
    response = agent.answer_question("What is total revenue?", use_claude=False)
    assert response["evidence_check"]["passed"] is True
    assert response["answer"] is not None


def test_orchestrator_intent_routes_to_correct_result():
    df = load_sales_file(SAMPLE)
    agent = SalesAnalystAgent(df)
    result = agent.get_analysis("product_analysis")
    assert result is not None
    assert "revenue" in result.columns
