## 🔧 AWS Bedrock Claude 설정

### 1단계: AWS CLI 설치 및 설정
```bash
# AWS CLI 설치
pip install awscli

# AWS 인증 설정
aws configure
# AWS Access Key ID: [입력]
# AWS Secret Access Key: [입력]
# Default region name: us-east-1
# Default output format: json
```

### 2단계: Bedrock 모델 액세스 요청

AWS Console에서 Claude 모델 액세스 권한을 요청해야 합니다:

1. [AWS Bedrock Console](https://console.aws.amazon.com/bedrock/) 접속
2. 왼쪽 메뉴에서 **"Model access"** 클릭
3. **"Modify model access"** 버튼 클릭
4. **Anthropic** 섹션에서 Claude 모델 체크
   - Claude 3.5 Sonnet v2 (권장)
   - Claude 3.5 Sonnet
   - Claude 3 Opus
   - Claude 3 Sonnet
   - Claude 3 Haiku
5. **"Save changes"** 클릭
6. 승인까지 몇 분 소요 (보통 즉시 승인)

### 3단계: IAM 권한 확인

AWS 사용자/역할에 Bedrock 권한이 있어야 합니다:
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream"
      ],
      "Resource": "arn:aws:bedrock:*::foundation-model/anthropic.claude-*"
    }
  ]
}
```

### 사용 가능한 Claude 모델

#### Cross-Region 모델 (권장 ⭐)

모든 리전에서 사용 가능하며 자동으로 최적의 리전으로 라우팅됩니다:

| 모델 ID | 설명 | 비용 |
|---------|------|------|
| `us.anthropic.claude-3-5-sonnet-20241022-v2:0` | Claude 3.5 Sonnet v2 (최신) | $3/$15 per MTok |
| `us.anthropic.claude-3-5-sonnet-20240620-v1:0` | Claude 3.5 Sonnet v1 | $3/$15 per MTok |

#### On-Demand 모델 (리전별 제한)

특정 리전에서만 사용 가능:

| 모델 ID | 설명 | 비용 |
|---------|------|------|
| `anthropic.claude-3-5-sonnet-20240620-v1:0` | Claude 3.5 Sonnet | $3/$15 per MTok |
| `anthropic.claude-3-opus-20240229-v1:0` | Claude 3 Opus | $15/$75 per MTok |
| `anthropic.claude-3-sonnet-20240229-v1:0` | Claude 3 Sonnet | $3/$15 per MTok |
| `anthropic.claude-3-haiku-20240307-v1:0` | Claude 3 Haiku | $0.25/$1.25 per MTok |

**권장 모델**: `us.anthropic.claude-3-5-sonnet-20241022-v2:0` (Cross-region, 최신)

### 사용 가능한 AWS 리전

Claude 3.5 Sonnet을 사용할 수 있는 리전:

- `us-east-1` (버지니아 북부) ⭐ 권장 - 모든 모델 지원
- `us-west-2` (오레곤)
- `ap-southeast-1` (싱가포르)
- `ap-northeast-1` (도쿄)
- `eu-central-1` (프랑크푸르트)
- `eu-west-3` (파리)

리전별 모델 가용성 확인:
```bash
aws bedrock list-foundation-models \
  --region us-east-1 \
  --by-provider anthropic \
  --query 'modelSummaries[*].[modelId,modelName]' \
  --output table
```

### 환경 설정 방법

#### 방법 1: AWS CLI (권장)
```bash
aws configure
```

#### 방법 2: 환경변수
```bash
export AWS_ACCESS_KEY_ID=xxxxx
export AWS_SECRET_ACCESS_KEY=xxxxx
export AWS_REGION=us-east-1
export BEDROCK_CLAUDE_MODEL=us.anthropic.claude-3-5-sonnet-20241022-v2:0
```

#### 방법 3: .env 파일
```env
AWS_REGION=us-east-1
BEDROCK_CLAUDE_MODEL=us.anthropic.claude-3-5-sonnet-20241022-v2:0
```

### 테스트
```bash
# 설정 확인
python -c "import boto3; print(boto3.Session().get_credentials())"

# 모델 액세스 확인
aws bedrock list-foundation-models --region us-east-1 --by-provider anthropic

# 프로그램 실행
python main.py
# 선택: 3. AWS Bedrock Claude
```

### 문제 해결

#### Error: "ValidationException"
- 원인: 모델 ID가 잘못되었거나 해당 리전에서 사용 불가
- 해결: 모델 ID 확인 및 리전 변경
```bash
  export AWS_REGION=us-east-1
  export BEDROCK_CLAUDE_MODEL=us.anthropic.claude-3-5-sonnet-20241022-v2:0
```

#### Error: "AccessDeniedException"
- 원인: Bedrock 모델 액세스 권한 없음
- 해결: AWS Console → Bedrock → Model access → Request access

#### Error: "ResourceNotFoundException"
- 원인: 모델이 해당 리전에 없음
- 해결: Cross-region 모델 사용 (`us.` 접두사)

### 비용 최적화

1. **Cross-region 모델 사용** (`us.` 접두사)
   - 자동 라우팅으로 지연 시간 감소
   - 모든 리전에서 동일한 가격

2. **적절한 모델 선택**
   - 간단한 작업: Claude 3 Haiku ($0.25/$1.25)
   - 균형잡힌 성능: Claude 3.5 Sonnet ($3/$15) ⭐ 권장
   - 최고 성능: Claude 3 Opus ($15/$75)

3. **토큰 사용 최적화**
   - `max_tokens` 설정 최적화
   - 불필요한 반복 호출 최소화