"""Schema validation utilities for CSV inputs."""

from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd


@dataclass
class ValidationResult:
    errors: list[str]
    warnings: list[str]

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)


def validate_dataframe(
    df: pd.DataFrame,
    name: str,
    required_columns: Iterable[str],
    date_columns: Sequence[str] | None = None,
    numeric_columns: Sequence[str] | None = None,
) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []

    required_set = set(required_columns)
    missing = sorted(required_set - set(df.columns))
    if missing:
        errors.append(f"{name}: missing columns {missing}")

    if df.empty:
        errors.append(f"{name}: dataset is empty")

    if not missing and date_columns:
        for column in date_columns:
            if column not in df.columns:
                continue
            invalid_count = int(df[column].isna().sum())
            if invalid_count:
                warnings.append(
                    f"{name}: {invalid_count} invalid or missing dates in '{column}'"
                )

    if not missing and numeric_columns:
        for column in numeric_columns:
            if column not in df.columns:
                continue
            non_numeric = (
                pd.to_numeric(df[column], errors="coerce").isna() & df[column].notna()
            )
            invalid_count = int(non_numeric.sum())
            if invalid_count:
                warnings.append(
                    f"{name}: {invalid_count} non-numeric values in '{column}'"
                )

    return ValidationResult(errors=errors, warnings=warnings)
