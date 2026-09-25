# Deploy chatbot_web to AWS Bedrock AgentCore Runtime (Docker / ECR)
# Run from the chatbot_web directory:
#   cd chatbot_web
#   .\deploy_agentcore.ps1 -AwsAccountId "" -Region "ap-south-1"

param (
    [Parameter(Mandatory=$true)]
    [string]$AwsAccountId,

    [Parameter(Mandatory=$false)]
    [string]$Region = "ap-south-1",

    [Parameter(Mandatory=$false)]
    [string]$RepoName = "asl-web-chatbot",

    [Parameter(Mandatory=$false)]
    [string]$ImageTag = "v1.0.1-arm64",

    [Parameter(Mandatory=$false)]
    [string]$RuntimeName = "Asl_Web_chatbot_runtime",

    [Parameter(Mandatory=$false)]
    [string]$ExecutionRoleName = "AgentCoreRuntimeExecutionRole",

    [Parameter(Mandatory=$false)]
    [string]$WebToken = "dev-token-change-in-prod"
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "======================================================="
Write-Host "  Deploying chatbot_web to AgentCore Runtime via ECR"
Write-Host "======================================================="
Write-Host ""

$ECR_REGISTRY = "$AwsAccountId.dkr.ecr.$Region.amazonaws.com"
$IMAGE_URI    = "$ECR_REGISTRY/$RepoName`:$ImageTag"
$ROLE_ARN     = "arn:aws:iam::$AwsAccountId`:role/$ExecutionRoleName"

Write-Host "Account  : $AwsAccountId"
Write-Host "Region   : $Region"
Write-Host "Image    : $IMAGE_URI"
Write-Host "Role     : $ROLE_ARN"
Write-Host "Runtime  : $RuntimeName"
Write-Host ""

# STEP 1 - Verify AWS credentials
Write-Host "[1/4] Verifying AWS credentials..."
$callerJson = aws sts get-caller-identity --region $Region
$caller = $callerJson | ConvertFrom-Json
Write-Host "  Arn     : $($caller.Arn)"
Write-Host "  Account : $($caller.Account)"
if ($caller.Account -ne $AwsAccountId) {
    Write-Host "ERROR: credential account $($caller.Account) does not match $AwsAccountId"
    exit 1
}
Write-Host "  OK"
Write-Host ""

# STEP 2 - ECR repo + docker login
Write-Host "[2/4] Authenticating with ECR..."
$repoExists = $false
try {
    $null = aws ecr describe-repositories --repository-names $RepoName --region $Region 2>&1
    if ($LASTEXITCODE -eq 0) { $repoExists = $true }
} catch {
    $repoExists = $false
}

if (-not $repoExists) {
    Write-Host "  Repo not found - creating $RepoName ..."
    aws ecr create-repository --repository-name $RepoName --region $Region | Out-Null
    Write-Host "  Repo created."
} else {
    Write-Host "  Repo $RepoName already exists."
}

aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $ECR_REGISTRY
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: docker login to ECR failed."
    exit 1
}
Write-Host "  ECR login OK."
Write-Host ""

# STEP 3 - Docker build and push
Write-Host "[3/4] Building Docker image for linux/arm64 (AgentCore requirement)..."
docker build --platform linux/arm64 -t $IMAGE_URI .
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: docker build failed."
    exit 1
}
Write-Host "  Build done."

Write-Host "  Pushing to ECR..."
docker push $IMAGE_URI
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: docker push failed."
    exit 1
}
Write-Host "  Pushed: $IMAGE_URI"
Write-Host ""

# STEP 4 - Create or update AgentCore Runtime (via boto3 — AWS CLI does not support this yet)
Write-Host "[4/4] Deploying AgentCore Runtime..."

$output = python _create_runtime.py $AwsAccountId $Region $IMAGE_URI $RuntimeName $ROLE_ARN $WebToken
Write-Host $output

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: AgentCore Runtime deploy failed."
    exit 1
}

Write-Host ""
Write-Host "======================================================="
Write-Host "  Deployment complete!"
Write-Host "  Runtime : $RuntimeName"
Write-Host "  Image   : $IMAGE_URI"
Write-Host "======================================================="
Write-Host ""
