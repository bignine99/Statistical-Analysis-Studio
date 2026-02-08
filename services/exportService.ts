
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import { jsPDF } from "jspdf";
import "jspdf-autotable";
import * as XLSX from "xlsx";
import { saveAs } from "file-saver";
import { Dataset, StatisticalInsight } from "../types";

export const exportToPDF = (insights: StatisticalInsight[], datasetName: string) => {
    const doc = new jsPDF();

    doc.setFontSize(22);
    doc.text("STAT-AGENT Russell Analysis Report", 20, 20);

    doc.setFontSize(12);
    doc.text(`Dataset: ${datasetName}`, 20, 30);
    doc.text(`Generated on: ${new Date().toLocaleString()}`, 20, 36);

    let yPos = 50;

    insights.forEach((insight, idx) => {
        if (yPos > 240) {
            doc.addPage();
            yPos = 20;
        }

        doc.setFontSize(14);
        doc.setTextColor(0, 100, 0);
        doc.text(`Card ${idx + 1}: ${insight.task_title}`, 20, yPos);
        yPos += 10;

        doc.setFontSize(10);
        doc.setTextColor(0, 0, 0);
        const splitFinding = doc.splitTextToSize(`Finding: ${insight.finding}`, 170);
        doc.text(splitFinding, 20, yPos);
        yPos += (splitFinding.length * 5) + 5;

        const splitInterpretation = doc.splitTextToSize(`Interpretation: ${insight.interpretation}`, 170);
        doc.text(splitInterpretation, 20, yPos);
        yPos += (splitInterpretation.length * 5) + 15;
    });

    doc.save(`Russell_Report_${datasetName.split('.')[0]}.pdf`);
};

export const exportToExcel = (dataset: Dataset, insights: StatisticalInsight[]) => {
    const wb = XLSX.utils.book_new();

    // Sheet 1: Metadata & Insights
    const insightData = insights.map(i => ({
        "Task": i.task_title,
        "Finding": i.finding,
        "Significance": i.statistical_significance,
        "Confidence": i.confidence
    }));
    const wsInsights = XLSX.utils.json_to_sheet(insightData);
    XLSX.utils.book_append_sheet(wb, wsInsights, "Executive Summary");

    // Sheet 2: Raw Data (Subset or Full)
    const wsData = XLSX.utils.json_to_sheet(dataset.rows.slice(0, 1000));
    XLSX.utils.book_append_sheet(wb, wsData, "Source Data (Top 1000)");

    const excelBuffer = XLSX.write(wb, { bookType: "xlsx", type: "array" });
    const data = new Blob([excelBuffer], { type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" });
    saveAs(data, `Russell_Data_${dataset.fileName.split('.')[0]}.xlsx`);
};
