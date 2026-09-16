from __future__ import annotations
import pandas as pd

from src.analytics.engine import run_analytics
from src.ai.intents import classify_intent, INTENT_TO_RESULT_KEY
from src.ai.prompts import SYSTEM_PROMPT, EXPLANATION_PROMPT_TEMPLATE, ROOT_CAUSE_PROMPT_TEMPLATE
from src.ai.claude_client import ask_claude
from src.qa.evidence import build_evidence, validate_evidence
from src.qa.analytical_qa import validate_kpis


class SalesAnalystAgent:
    """
    USER QUESTION -> intent classification -> analytics lookup -> evidence
    validation -> Claude interpretation. Claude is never given the raw
    dataframe; it only ever sees the JSON-serializable evidence this class
    assembles, which keeps every answer traceable back to a deterministic
    calculation.
    """

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        self.analytics = run_analytics(dataframe)

    def available_data(self) -> dict:
        return {k: (v is not None) for k, v in self.analytics.items()}

    def get_analysis(self, intent: str):
        key = INTENT_TO_RESULT_KEY.get(intent)
        if not key:
            return {"error": f"Unknown intent: {intent}"}
        return self.analytics.get(key)

    def answer_question(self, question: str, use_claude: bool = True) -> dict:
        intent = classify_intent(question)
        result = self.get_analysis(intent)

        evidence = build_evidence(
            metric=intent,
            value=_summarize_for_evidence(result),
        )
        evidence_check = validate_evidence(evidence)

        response = {
            "question": question,
            "intent": intent,
            "evidence": evidence,
            "evidence_check": evidence_check,
        }

        if not evidence_check["passed"]:
            response["answer"] = (
                "The available data is insufficient to answer this question "
                "with the required evidence."
            )
            return response

        if use_claude:
            prompt = EXPLANATION_PROMPT_TEMPLATE.format(
                question=question, code=f"run_analytics()['{INTENT_TO_RESULT_KEY.get(intent)}']",
                result=evidence["value"],
            )
            response["answer"] = ask_claude(SYSTEM_PROMPT, prompt)
        else:
            response["answer"] = str(evidence["value"])

        return response

    def root_cause_analysis(self, question: str) -> str:
        """The flagship 'Why did sales fall?' style query."""
        a = self.analytics
        prompt = ROOT_CAUSE_PROMPT_TEMPLATE.format(
            question=question,
            overall_kpis=a.get("core_kpis"),
            monthly_growth=_df_tail_dict(a.get("monthly_growth")),
            products=_df_head_dict(a.get("products")),
            categories=_df_head_dict(a.get("categories")),
            customers=_df_head_dict(a.get("customers")),
            stores=_df_head_dict(a.get("stores")),
            regions=_df_head_dict(a.get("regions")),
            anomalies=a.get("anomalies"),
            targets=a.get("target_performance"),
        )
        return ask_claude(SYSTEM_PROMPT, prompt, max_tokens=1500)


def _summarize_for_evidence(result):
    if result is None:
        return None
    if isinstance(result, pd.DataFrame):
        return result.head(10).to_dict("records")
    return result


def _df_head_dict(df, n=10):
    if df is None or not isinstance(df, pd.DataFrame):
        return "Not available"
    return df.head(n).to_dict("records")


def _df_tail_dict(df, n=6):
    if df is None or not isinstance(df, pd.DataFrame):
        return "Not available"
    return df.tail(n).to_dict("records")
