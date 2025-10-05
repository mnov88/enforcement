import importlib.util
from pathlib import Path

import pandas as pd

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "4_enrich_prepare_outputs.py"
spec = importlib.util.spec_from_file_location("phase4_enrichment", MODULE_PATH)
phase4 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(phase4)


def test_compute_art5_and_rights_handles_profiles_and_gaps():
    df = pd.DataFrame(
        [
            {
                "id": "CASE1",
                "a21_art5_lawfulness_fairness": "BREACHED",
                "a22_art5_purpose_limitation": "NO",
                "a23_art5_data_minimization": "NO",
                "a24_art5_accuracy": "NO",
                "a25_art5_storage_limitation": "NO",
                "a26_art5_integrity_confidentiality": "NO",
                "a27_art5_accountability": "NO",
                "a37_right_access_violated": "YES",
                "a38_right_rectification_violated": "NO",
                "a39_right_erasure_violated": "NOT_DISCUSSED",
                "a40_right_restriction_violated": "NO",
                "a41_right_portability_violated": "NO",
                "a42_right_object_violated": "NO",
                "a43_transparency_violated": "NO",
                "a44_automated_decisions_violated": "NO",
            },
            {
                "id": "CASE2",
                "a21_art5_lawfulness_fairness": "BREACHED",
                "a22_art5_purpose_limitation": "BREACHED",
                "a23_art5_data_minimization": "NO",
                "a24_art5_accuracy": "NO",
                "a25_art5_storage_limitation": "NO",
                "a26_art5_integrity_confidentiality": "NO",
                "a27_art5_accountability": "NO",
                "a37_right_access_violated": "YES",
                "a38_right_rectification_violated": "YES",
                "a39_right_erasure_violated": "NO",
                "a40_right_restriction_violated": "NO",
                "a41_right_portability_violated": "NO",
                "a42_right_object_violated": "NO",
                "a43_transparency_violated": "NO",
                "a44_automated_decisions_violated": "NO",
            },
        ]
    )

    articles_df = pd.DataFrame(
        [
            {
                "id": "CASE1",
                "article_reference": "Art. 5(1)(a)",
                "article_reference_detail": "5(1)(a)",
                "article_detail_tokens": "1;a",
                "article_label": "5",
                "article_number": 5,
                "position": 1,
            },
            {
                "id": "CASE1",
                "article_reference": "Art. 15",
                "article_reference_detail": "15",
                "article_detail_tokens": "",
                "article_label": "15",
                "article_number": 15,
                "position": 2,
            },
            {
                "id": "CASE2",
                "article_reference": "Art. 6",
                "article_reference_detail": "6",
                "article_detail_tokens": "",
                "article_label": "6",
                "article_number": 6,
                "position": 1,
            },
        ]
    )

    result = phase4.compute_art5_and_rights(df.copy(), articles_df)

    assert result.loc[result["id"] == "CASE1", "rights_violated_count"].iloc[0] == 1
    assert result.loc[result["id"] == "CASE1", "rights_profile"].iloc[0] == "A15"
    assert bool(result.loc[result["id"] == "CASE1", "flag_articles_vs_rights_gap"].iloc[0]) is False

    assert result.loc[result["id"] == "CASE2", "rights_violated_count"].iloc[0] == 2
    assert result.loc[result["id"] == "CASE2", "rights_profile"].iloc[0] == "A15;A16"
    assert bool(result.loc[result["id"] == "CASE2", "flag_articles_vs_rights_gap"].iloc[0]) is True
    assert bool(result.loc[result["id"] == "CASE2", "breach_has_art5"].iloc[0]) is True
    assert bool(
        result.loc[result["id"] == "CASE2", "flag_art5_breached_but_5_not_in_77"].iloc[0]
    ) is True


def test_compute_measures_rolls_up_flags():
    df = pd.DataFrame(
        [
            {
                "id": "CASE1",
                "a45_warning_issued": "YES",
                "a46_reprimand_issued": "NO",
                "a47_comply_data_subject_order": "NO",
                "a48_compliance_order": "NO",
                "a49_breach_communication_order": "NO",
                "a50_erasure_restriction_order": "NO",
                "a51_certification_withdrawal": "NO",
                "a52_data_flow_suspension": "NO",
                "a53_fine_imposed": "NO",
            },
            {
                "id": "CASE2",
                "a45_warning_issued": "NO",
                "a46_reprimand_issued": "YES",
                "a47_comply_data_subject_order": "NO",
                "a48_compliance_order": "YES",
                "a49_breach_communication_order": "NO",
                "a50_erasure_restriction_order": "NO",
                "a51_certification_withdrawal": "NO",
                "a52_data_flow_suspension": "NO",
                "a53_fine_imposed": "YES",
            },
        ]
    )
    df["fine_imposed_bool"] = df["a53_fine_imposed"].apply(phase4.yes_no_to_bool)

    result = phase4.compute_measures(df.copy())

    case1 = result[result["id"] == "CASE1"].iloc[0]
    assert bool(case1["measure_any_bool"]) is True
    assert case1["measure_count"] == 1
    assert case1["sanction_profile"] == "Warning"
    assert bool(case1["is_warning_only"]) is True
    assert bool(case1["is_fine_only"]) is False

    case2 = result[result["id"] == "CASE2"].iloc[0]
    assert bool(case2["measure_any_bool"]) is True
    assert case2["measure_count"] == 2
    assert case2["sanction_profile"] == "ComplianceOrder"
    assert bool(case2["is_warning_only"]) is False
    assert bool(case2["is_fine_only"]) is False


def test_build_graph_exports_writes_expected_nodes_and_edges(tmp_path):
    enriched_df = pd.DataFrame(
        [
            {
                "id": "CASE1",
                "decision_year": 2022,
                "country_code": "DE",
                "fine_amount_eur": 1000.0,
                "sanction_profile": "Warning",
                "authority_name_norm": "Bavarian DPA",
                "a7_defendant_name": "Acme GmbH",
                "a8_defendant_class": "PRIVATE",
                "a9_enterprise_size": "SME",
                "a12_sector": "TECH",
            },
            {
                "id": "CASE2",
                "decision_year": 2023,
                "country_code": "FR",
                "fine_amount_eur": 2500.0,
                "sanction_profile": "ComplianceOrder",
                "authority_name_norm": "Bavarian DPA",
                "a7_defendant_name": "Acme GmbH",
                "a8_defendant_class": "PRIVATE",
                "a9_enterprise_size": "SME",
                "a12_sector": "TECH",
            },
        ]
    )

    contexts_df = pd.DataFrame(
        [
            {"id": "CASE1", "processing_context": "CCTV", "position": 1},
            {"id": "CASE1", "processing_context": "CCTV", "position": 2},
            {"id": "CASE2", "processing_context": "HR", "position": 1},
        ]
    )

    guidelines_df = pd.DataFrame(
        [
            {"id": "CASE1", "guideline": "WP29", "position": 1},
            {"id": "CASE2", "guideline": "EDPB", "position": 1},
        ]
    )

    articles_df = pd.DataFrame(
        [
            {
                "id": "CASE1",
                "article_reference": "Art. 5",
                "article_reference_detail": "5",
                "article_detail_tokens": "",
                "article_label": "5",
                "article_number": 5,
                "position": 1,
            },
            {
                "id": "CASE2",
                "article_reference": "Art. 6",
                "article_reference_detail": "6",
                "article_detail_tokens": "",
                "article_label": "6",
                "article_number": 6,
                "position": 1,
            },
        ]
    )

    phase4.build_graph_exports(enriched_df, contexts_df, guidelines_df, articles_df, tmp_path)

    graph_dir = tmp_path / "graph"

    decision_nodes = pd.read_csv(graph_dir / "nodes_decision.csv").sort_values("id").reset_index(drop=True)
    assert decision_nodes["id:ID"].tolist() == ["Decision|CASE1", "Decision|CASE2"]

    authority_nodes = pd.read_csv(graph_dir / "nodes_authority.csv")
    assert authority_nodes["id:ID"].unique().tolist() == ["Authority|BAVARIAN_DPA"]

    edges_authority = pd.read_csv(graph_dir / "edges_decision_authority.csv")
    assert len(edges_authority) == 2
    assert edges_authority[":END_ID"].unique().tolist() == ["Authority|BAVARIAN_DPA"]

    edges_context = pd.read_csv(graph_dir / "edges_decision_context.csv")
    assert len(edges_context) == 2
    assert set(edges_context[":END_ID"]) == {"Context|CCTV", "Context|HR"}

    edges_article = pd.read_csv(graph_dir / "edges_decision_article.csv")
    assert len(edges_article) == 2
    assert set(edges_article[":END_ID"]) == {"Article|5", "Article|6"}

    edges_guideline = pd.read_csv(graph_dir / "edges_decision_guideline.csv")
    assert set(edges_guideline[":END_ID"]) == {"Guideline|WP29", "Guideline|EDPB"}
