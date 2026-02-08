# 📊 Statistical Analysis Studio - 네이버 클라우드 배포 가이드

## 프로젝트 개요

**Statistical Analysis Studio**는 CSV 데이터를 업로드하면 AI가 통계 분석을 수행하는 웹 애플리케이션입니다.

- **GitHub**: https://github.com/bignine99/Statistical-Analysis-Studio
- **기술 스택**: React + TypeScript + Vite (프론트엔드) / FastAPI + Python (백엔드)
- **AI 엔진**: Google Gemini API
- **통계 엔진**: Python statsmodels, scipy, plotly

---

## 아키텍처

```
[사용자 브라우저]
      │
      ▼
[Nginx 리버스 프록시]
      │
      ├── /stat/ → 프론트엔드 정적 파일 (빌드된 dist/)
      │
      └── /stat/api/ → FastAPI 백엔드 (localhost:8001)
                         ├── statsmodels (회귀분석, VIF)
                         ├── scipy (정규성 검정, 상관분석)
                         └── plotly (차트 생성)
```

### 프론트엔드 → 백엔드 통신 흐름
1. 사용자가 CSV 업로드 → 프론트엔드가 파일 파싱 (papaparse)
2. 사용자가 종속변수 선택 → "분석 시작" 클릭
3. 프론트엔드가 백엔드 API (`/stat/api/analyze`)로 데이터 전송
4. 백엔드가 통계 분석 수행 후 결과 + Plotly 차트 JSON 반환
5. Gemini API가 결과를 자연어로 해석 (프론트엔드에서 직접 호출)

---

## 배포 순서

### Step 1: 서버에서 소스코드 가져오기

```bash
# 적절한 디렉토리로 이동 (예: /home/cho/ 또는 기존 프로젝트 디렉토리)
cd /home/cho
git clone https://github.com/bignine99/Statistical-Analysis-Studio.git
cd Statistical-Analysis-Studio
```

### Step 2: 환경 변수 설정

프론트엔드 빌드 시 Gemini API 키가 필요합니다. (인증 모달에서 비밀번호 "0172"를 입력했을 때 사용되는 기본 키)

```bash
# .env.local 파일 생성 (이 파일은 git에 포함되어 있지 않음)
cat > .env.local << 'EOF'
GEMINI_API_KEY=AIzaSyCVfNJqCKiSz0Er4Xhcmuhnj1q2eD7E2kk
EOF
```

### Step 3: 프론트엔드 빌드

```bash
# Node.js 의존성 설치
npm install

# ⚠️ 중요: 빌드 전에 vite.config.ts의 base 경로 설정 필요
# Nginx에서 /stat/ 경로로 서빙할 경우:
```

**vite.config.ts를 수정해야 합니다:**

```typescript
import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '');
  return {
    base: '/stat/',  // ← 이 줄 추가! (Nginx 경로에 맞춰 설정)
    server: {
      port: 3000,
      host: '0.0.0.0',
    },
    plugins: [react()],
    define: {
      'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY),
      'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY)
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      }
    }
  };
});
```

```bash
# 프로덕션 빌드
npm run build
# → dist/ 폴더에 정적 파일 생성됨
```

### Step 4: 백엔드 설정

```bash
# Python 가상환경 생성 (권장)
cd backend
python3 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
# 패키지: fastapi, uvicorn, pandas, numpy, statsmodels, scipy, plotly, python-multipart
```

**⚠️ 백엔드 main.py에서 CORS 설정 확인 필요:**

`backend/main.py`에서 CORS가 허용되어 있는지 확인하세요. 프로덕션 도메인에 맞게 수정이 필요할 수 있습니다:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 실제 도메인으로 제한 권장
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

**⚠️ 프론트엔드의 백엔드 URL 수정 필요:**

`services/backendService.ts` 파일에서 `BACKEND_URL`을 서버 환경에 맞게 수정해야 합니다:

```typescript
// 개발 환경: http://localhost:8000
// 프로덕션 (Nginx 리버스 프록시 사용 시):
const BACKEND_URL = '/stat/api';
// 또는 절대 URL:
// const BACKEND_URL = 'https://yourdomain.com/stat/api';
```

이 값을 변경한 후 프론트엔드를 다시 빌드해야 합니다 (`npm run build`).

### Step 5: PM2로 백엔드 실행

```bash
# PM2 ecosystem 파일 생성
cat > /home/cho/Statistical-Analysis-Studio/ecosystem.config.js << 'EOF'
module.exports = {
  apps: [
    {
      name: 'stat-backend',
      cwd: '/home/cho/Statistical-Analysis-Studio/backend',
      script: 'venv/bin/uvicorn',
      args: 'main:app --host 127.0.0.1 --port 8001',
      interpreter: 'none',
      env: {
        PYTHONPATH: '/home/cho/Statistical-Analysis-Studio/backend'
      }
    }
  ]
};
EOF

# PM2로 백엔드 시작
pm2 start ecosystem.config.js
pm2 save
```

### Step 6: Nginx 설정

기존 Nginx 설정 파일에 다음 블록을 추가하세요:

```nginx
# Statistical Analysis Studio
location /stat/ {
    alias /home/cho/Statistical-Analysis-Studio/dist/;
    try_files $uri $uri/ /stat/index.html;
}

# 백엔드 API 프록시
location /stat/api/ {
    proxy_pass http://127.0.0.1:8001/;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection 'upgrade';
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_read_timeout 300s;  # 통계 분석이 오래 걸릴 수 있으므로 타임아웃 여유있게
    proxy_send_timeout 300s;
    client_max_body_size 50M;  # CSV 파일 업로드 크기 제한
}
```

```bash
# Nginx 설정 테스트 및 재시작
sudo nginx -t
sudo systemctl reload nginx
```

---

## 주요 파일 설명

| 파일/폴더 | 역할 |
|---|---|
| `App.tsx` | 메인 앱 컴포넌트 (약 2,400줄). 랜딩페이지, 인증모달, 대시보드, 분석 로직 모두 포함 |
| `index.html` | HTML 진입점 |
| `index.tsx` | React 진입점 |
| `index.css` | 전역 CSS 스타일 |
| `vite.config.ts` | Vite 빌드 설정 (base path, 환경변수 주입) |
| `services/backendService.ts` | Python 백엔드 API 호출 로직. **BACKEND_URL을 배포 환경에 맞게 수정 필요** |
| `services/geminiService.ts` | Gemini API 직접 호출 (프론트엔드에서) |
| `services/exportService.ts` | PDF/Excel 내보내기 기능 |
| `services/dataProcessor.ts` | CSV 데이터 전처리 |
| `components/PlotlyChart.tsx` | Plotly 차트 렌더링 컴포넌트 |
| `components/LogTerminal.tsx` | 분석 로그 표시 터미널 |
| `backend/main.py` | FastAPI 서버 (약 78,000줄). 통계 분석 엔드포인트 (/analyze) |
| `backend/requirements.txt` | Python 의존성 목록 |
| `.env.local` | Gemini API 키 (git에 미포함, 서버에서 직접 생성 필요) |
| `types.ts` | TypeScript 타입 정의 |
| `constants.ts` | 앱 전체 상수 |

---

## 인증 시스템

앱에는 간단한 인증 모달이 있습니다:

1. **비밀번호 모드**: "0172" 입력 시 → 기본 API 키(`.env.local`의 키)로 접근
2. **API 키 모드**: 사용자가 직접 Gemini API 키를 입력

이 인증은 서버 사이드가 아닌 **클라이언트 사이드**에서만 동작합니다.

---

## 배포 후 확인 사항

1. **프론트엔드 접속**: `https://yourdomain.com/stat/` 에서 랜딩 페이지가 보이는지 확인
2. **인증 모달**: "분석 시작" 클릭 → 모달 팝업 → 비밀번호 "0172" 입력
3. **CSV 업로드**: 대시보드에서 CSV 파일 업로드 가능한지 확인
4. **백엔드 연결**: 종속변수 선택 → "분석 시작" 클릭 → 결과가 나오는지 확인
5. **PM2 상태**: `pm2 status` 로 stat-backend가 online인지 확인
6. **에러 로그**: `pm2 logs stat-backend` 로 백엔드 에러 확인

---

## 트러블슈팅

### 프론트엔드가 로드되지 않을 때
- Nginx의 `alias` 경로가 `dist/` 폴더를 정확히 가리키는지 확인
- `npm run build`가 성공적으로 완료되었는지 확인
- `vite.config.ts`의 `base` 값이 Nginx location과 일치하는지 확인

### 백엔드 API 오류
- `pm2 logs stat-backend` 로 에러 확인
- Python 가상환경이 활성화되었는지 확인
- `pip install -r requirements.txt` 가 완전히 설치되었는지 확인
- statsmodels, scipy 등은 빌드에 시간이 걸릴 수 있음

### CORS 에러
- `backend/main.py`에서 CORS 미들웨어 설정 확인
- Nginx에서 프록시 설정이 올바른지 확인

### 분석이 시작되지 않을 때
- `services/backendService.ts`의 `BACKEND_URL` 값 확인
- 브라우저 개발자 도구(F12) → Network 탭에서 API 요청 URL 확인
- 백엔드 포트(8001)가 정상적으로 리스닝 중인지 확인: `netstat -tlnp | grep 8001`

---

## 요약: 핵심 수정 포인트 (배포 시 반드시 변경)

1. **`vite.config.ts`** → `base: '/stat/'` 추가
2. **`services/backendService.ts`** → `BACKEND_URL`을 `/stat/api` 또는 실제 도메인으로 변경
3. **`.env.local`** → 서버에 직접 생성 (Gemini API Key)
4. **`backend/main.py`** → CORS 도메인 설정 확인
5. **Nginx** → location 블록 추가
6. **PM2** → 백엔드 프로세스 등록

이 6개 포인트만 정확히 설정하면 배포가 완료됩니다.
