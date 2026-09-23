import csv
import hashlib
from pathlib import Path
from zipfile import ZipFile

import pytest

from deep_recsys_lifecycle.fixture import load_movielens_fixture
from deep_recsys_lifecycle.kafka import InMemoryKafkaBoundary
from deep_recsys_lifecycle.lightgcn import LightGCNConfig
from deep_recsys_lifecycle.models import RatingEvent
from deep_recsys_lifecycle.movielens import prepare_movielens_20m
from deep_recsys_lifecycle.multivae import MultVAEConfig
from deep_recsys_lifecycle.rolling import run_rolling_lifecycle


def _archive(path: Path) -> str:
    rows = [
        (2, 11, 4.0, 5),
        (1, 10, 3.5, 1),
        (1, 12, 5.0, 5),
    ]
    csv_path = path.parent / "ratings.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.writer(destination)
        writer.writerow(("userId", "movieId", "rating", "timestamp"))
        writer.writerows(rows)
    with ZipFile(path, "w") as bundle:
        bundle.write(csv_path, "ml-20m/ratings.csv")
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def test_prepared_source_checks_archive_and_preserves_canonical_chronology(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "ml-20m.zip"
    expected_md5 = _archive(archive)
    source = prepare_movielens_20m(
        tmp_path / "cache", archive_path=archive, expected_md5=expected_md5,
        expected_rating_count=3,
    )
    expected = tuple(sorted(
        (
            RatingEvent.from_movielens(2, 11, 4.0, 5),
            RatingEvent.from_movielens(1, 10, 3.5, 1),
            RatingEvent.from_movielens(1, 12, 5.0, 5),
        ),
        key=lambda event: (event.event_time, event.event_id),
    ))

    assert len(source) == 3
    assert tuple(source) == expected
    assert tuple(source[1:]) == expected[1:]
    assert source[-1] == expected[-1]
    assert source.dataset_checksum == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert tuple(prepare_movielens_20m(
        tmp_path / "cache", archive_path=archive, expected_md5=expected_md5,
        expected_rating_count=3,
    )) == expected


def test_prepared_source_rejects_archive_checksum_mismatch(tmp_path: Path) -> None:
    archive = tmp_path / "ml-20m.zip"
    _archive(archive)
    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_movielens_20m(tmp_path / "cache", archive_path=archive, expected_md5="0" * 32)


def test_prepared_source_runs_same_six_stage_lifecycle_contract(tmp_path: Path) -> None:
    archive = tmp_path / "ml-20m.zip"
    fixture_path = Path(__file__).parents[1] / "src/deep_recsys_lifecycle/fixtures/ratings.csv"
    with ZipFile(archive, "w") as bundle:
        bundle.write(fixture_path, "ml-20m/ratings.csv")
    checksum = hashlib.md5(archive.read_bytes(), usedforsecurity=False).hexdigest()
    source = prepare_movielens_20m(
        tmp_path / "cache", archive_path=archive, expected_md5=checksum,
        expected_rating_count=len(load_movielens_fixture()),
    )

    result = run_rolling_lifecycle(
        tmp_path / "full", InMemoryKafkaBoundary(), source_events=source,
        evaluation_cohort_limit=5_000,
        multivae_config=MultVAEConfig(epochs=1),
        lightgcn_config=LightGCNConfig(epochs=1),
    )

    assert len(result.stages) == 6
    assert result.stages[-1].active
    assert "MovieLens 20M" in result.report_path.read_text(encoding="utf-8")
    assert result.dataset_checksum == source.dataset_checksum
