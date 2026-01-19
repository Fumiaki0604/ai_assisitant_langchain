@echo off
REM Secrets Manager シークレット作成スクリプト (Windows)

setlocal enabledelayedexpansion

set APP_NAME=slack-ai-assistant
set AWS_REGION=us-west-2

echo ============================================
echo Secrets Manager セットアップ
echo ============================================
echo.

REM .envファイルから値を読み込み
if not exist "..\.env" (
    echo Error: ..\.env ファイルが見つかりません
    exit /b 1
)

for /f "tokens=1,2 delims==" %%a in ('type "..\.env" ^| findstr /v "^#"') do (
    set %%a=%%b
)

REM Pinecone API Key
echo Pinecone API Key を登録...
aws secretsmanager create-secret --name "%APP_NAME%/pinecone-api-key" --secret-string "%PINECONE_API_KEY%" --region %AWS_REGION% 2>nul || aws secretsmanager update-secret --secret-id "%APP_NAME%/pinecone-api-key" --secret-string "%PINECONE_API_KEY%" --region %AWS_REGION%

REM Slack Bot Token
echo Slack Bot Token を登録...
aws secretsmanager create-secret --name "%APP_NAME%/slack-bot-token" --secret-string "%SLACK_BOT_TOKEN%" --region %AWS_REGION% 2>nul || aws secretsmanager update-secret --secret-id "%APP_NAME%/slack-bot-token" --secret-string "%SLACK_BOT_TOKEN%" --region %AWS_REGION%

REM Slack App Token
echo Slack App Token を登録...
aws secretsmanager create-secret --name "%APP_NAME%/slack-app-token" --secret-string "%SLACK_APP_TOKEN%" --region %AWS_REGION% 2>nul || aws secretsmanager update-secret --secret-id "%APP_NAME%/slack-app-token" --secret-string "%SLACK_APP_TOKEN%" --region %AWS_REGION%

REM Slack Signing Secret
echo Slack Signing Secret を登録...
aws secretsmanager create-secret --name "%APP_NAME%/slack-signing-secret" --secret-string "%SLACK_SIGNING_SECRET%" --region %AWS_REGION% 2>nul || aws secretsmanager update-secret --secret-id "%APP_NAME%/slack-signing-secret" --secret-string "%SLACK_SIGNING_SECRET%" --region %AWS_REGION%

echo.
echo ============================================
echo シークレット登録完了!
echo ============================================
echo.

endlocal
