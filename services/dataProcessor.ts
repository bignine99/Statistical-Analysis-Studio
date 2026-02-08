
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import { Dataset, PreprocessingSummary } from "../types";

export const preprocessData = (dataset: Dataset): { cleanedDataset: Dataset; summary: PreprocessingSummary } => {
    let missingFixed = 0;
    let duplicatesRemoved = 0;

    // 1. Remove duplicates
    const seen = new Set();
    const uniqueRows = dataset.rows.filter(row => {
        const str = JSON.stringify(row);
        if (seen.has(str)) {
            duplicatesRemoved++;
            return false;
        }
        seen.add(str);
        return true;
    });

    // 2. Fix missing values (Mean imputation for numbers)
    const columns = dataset.headers;
    const colStats: any = {};

    columns.forEach(col => {
        const vals = uniqueRows.map(r => r[col]).filter(v => typeof v === 'number' && v !== null);
        if (vals.length > 0) {
            colStats[col] = vals.reduce((a, b) => a + b, 0) / vals.length;
        }
    });

    const cleanedRows = uniqueRows.map(row => {
        const newRow = { ...row };
        columns.forEach(col => {
            if (newRow[col] === null || newRow[col] === undefined || newRow[col] === '') {
                if (colStats[col] !== undefined) {
                    newRow[col] = colStats[col];
                    missingFixed++;
                }
            }
        });
        return newRow;
    });

    return {
        cleanedDataset: {
            ...dataset,
            rows: cleanedRows,
            rowCount: cleanedRows.length
        },
        summary: {
            missing_values_fixed: missingFixed,
            outliers_detected: 0, // Calculated separately
            duplicates_removed: duplicatesRemoved
        }
    };
};

export const detectOutliers = (dataset: Dataset, column: string) => {
    const values = dataset.rows.map(r => r[column]).filter(v => typeof v === 'number');
    if (values.length < 4) return [];

    const sorted = [...values].sort((a, b) => a - b);
    const q1 = sorted[Math.floor(sorted.length / 4)];
    const q3 = sorted[Math.floor(sorted.length * 0.75)];
    const iqr = q3 - q1;
    const lowerBound = q1 - 1.5 * iqr;
    const upperBound = q3 + 1.5 * iqr;

    return dataset.rows
        .map((row, idx) => ({ value: row[column], idx }))
        .filter(item => typeof item.value === 'number' && (item.value < lowerBound || item.value > upperBound))
        .map(item => ({
            column,
            row_index: item.idx,
            value: item.value,
            reason: item.value > upperBound ? 'Extreme High (IQR)' : 'Extreme Low (IQR)'
        }));
};
