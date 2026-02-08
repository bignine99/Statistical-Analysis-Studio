
/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
*/

import { GoogleGenAI, Type } from "@google/genai";
import { SYSTEM_INSTRUCTION_PLANNER, SYSTEM_INSTRUCTION_EXECUTOR } from "../constants";
import { Dataset, InvestigationTask, StatisticalInsight } from "../types";

const getClient = () => {
  const apiKey = process.env.API_KEY;
  if (!apiKey) throw new Error("API 키가 선택되지 않았습니다. 상단 버튼을 통해 키를 설정하십시오.");
  return new GoogleGenAI({ apiKey });
};

export const planInvestigations = async (dataset: Dataset, dependentVariable?: string): Promise<InvestigationTask[]> => {
  const ai = getClient();
  const prompt = `
  파일명: ${dataset.fileName}
  컬럼: ${dataset.headers.join(', ')}
  데이터 요약: ${JSON.stringify(dataset.rows.slice(0, 5))}
  ${dependentVariable ? `분석 목표: '${dependentVariable}'을(를) 종속변수(Target)로 설정하여 이 변수에 영향을 미치는 요인과 상관관계를 중점적으로 분석하세요.` : ''}
  
  이 데이터셋에서 통계적으로 유의미한 통찰을 얻기 위한 5가지 분석 계획을 세워주세요.
  `;

  const response = await ai.models.generateContent({
    model: 'gemini-3-pro-preview',
    contents: prompt,
    config: {
      systemInstruction: SYSTEM_INSTRUCTION_PLANNER,
      responseMimeType: 'application/json',
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          tasks: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                id: { type: Type.STRING },
                title: { type: Type.STRING },
                target_columns: { type: Type.ARRAY, items: { type: Type.STRING } },
                methodology: { type: Type.STRING }
              },
              required: ["id", "title", "target_columns", "methodology"]
            }
          }
        }
      }
    }
  });

  const res = JSON.parse(response.text.trim());
  return res.tasks.map((t: any) => ({ ...t, status: 'pending' }));
};

export const executeInvestigation = async (dataset: Dataset, task: InvestigationTask): Promise<StatisticalInsight> => {
  const ai = getClient();
  const relevantData = dataset.rows.slice(0, 30).map(row => {
    const subset: any = {};
    task.target_columns.forEach(col => subset[col] = row[col]);
    return subset;
  });

  const prompt = `
  작업: ${task.title}
  방법: ${task.methodology}
  샘플: ${JSON.stringify(relevantData)}
  
  위 데이터를 분석하고, 결과를 시각화할 수 있는 데이터 포인트를 함께 제공하세요.
  `;

  const response = await ai.models.generateContent({
    model: 'gemini-3-pro-preview',
    contents: prompt,
    config: {
      systemInstruction: SYSTEM_INSTRUCTION_EXECUTOR,
      responseMimeType: 'application/json',
      responseSchema: {
        type: Type.OBJECT,
        properties: {
          finding: { type: Type.STRING },
          statistical_significance: { type: Type.STRING },
          interpretation: { type: Type.STRING },
          recommended_viz: { type: Type.STRING },
          confidence: { type: Type.NUMBER },
          viz_type: { type: Type.STRING },
          viz_data: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                label: { type: Type.STRING },
                value: { type: Type.NUMBER }
              },
              required: ["label", "value"]
            }
          }
        },
        required: ["finding", "statistical_significance", "interpretation", "recommended_viz", "confidence", "viz_type", "viz_data"]
      }
    }
  });

  const res = JSON.parse(response.text.trim());
  return {
    id: Math.random().toString(36).substring(7),
    task_title: task.title,
    ...res
  };
};
