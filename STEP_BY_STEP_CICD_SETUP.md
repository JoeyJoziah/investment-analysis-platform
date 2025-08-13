# 🚀 Step-by-Step CI/CD Setup Walkthrough
**Your Personal Guide to Setting Up the Investment Analysis App CI/CD Pipeline**

---

## 📋 Pre-Setup Checklist

Before we begin, ensure you have:
- [ ] GitHub repository created for the project
- [ ] Admin access to the repository
- [ ] Docker Hub account (free)
- [ ] DigitalOcean account (or AWS)
- [ ] Slack workspace (optional)
- [ ] API keys ready (we'll get these if you don't have them)

---

## 🔐 STEP 1: Configure GitHub Secrets

### 1.1 Navigate to Secrets Page

1. **Open your GitHub repository** in a browser
2. **Click on "Settings"** (gear icon in the top menu bar)
3. **Scroll down** in the left sidebar
4. **Click on "Secrets and variables"**
5. **Click on "Actions"**
6. You'll see a page titled "Actions secrets and variables"

### 1.2 Add Required Secrets

Click the green **"New repository secret"** button for each secret below:

#### 🔴 CRITICAL SECRETS (Add These First)

##### DATABASE_URL
- **Name:** `DATABASE_URL`
- **Value Example:** 
  ```
  postgresql://postgres:yourpassword@localhost:5432/investment_db
  ```
- **How to get it:**
  - If using local: Use the example above with your password
  - If using cloud: Get from your cloud provider's database dashboard
- **Test it:** Run in terminal:
  ```bash
  psql "postgresql://postgres:yourpassword@localhost:5432/investment_db" -c "SELECT 1"
  ```

##### REDIS_URL
- **Name:** `REDIS_URL`
- **Value Example:**
  ```
  redis://localhost:6379
  ```
- **How to get it:**
  - Local: Use `redis://localhost:6379`
  - Cloud: Get from your Redis provider
- **Test it:** Run:
  ```bash
  redis-cli -u "redis://localhost:6379" ping
  ```

#### 🟡 API KEYS (Get These from Providers)

##### ALPHA_VANTAGE_API_KEY
- **Name:** `ALPHA_VANTAGE_API_KEY`
- **How to get it:**
  1. Go to https://www.alphavantage.co/support/#api-key
  2. Enter your email
  3. Click "GET FREE API KEY"
  4. Copy the key from your email
- **Value:** Your API key (looks like: `ABCD1234EFGH5678`)
- **Free Tier:** 25 API calls/day, 5 calls/minute

##### FINNHUB_API_KEY
- **Name:** `FINNHUB_API_KEY`
- **How to get it:**
  1. Go to https://finnhub.io/
  2. Click "Get free API key"
  3. Sign up with email
  4. Go to Dashboard → API Key
- **Value:** Your API key (looks like: `ct0abc123def456ghi789`)
- **Free Tier:** 60 calls/minute

##### POLYGON_API_KEY
- **Name:** `POLYGON_API_KEY`
- **How to get it:**
  1. Go to https://polygon.io/
  2. Click "Get your Free API Key"
  3. Sign up
  4. Find key in Dashboard
- **Value:** Your API key
- **Free Tier:** 5 calls/minute

##### NEWS_API_KEY
- **Name:** `NEWS_API_KEY`
- **How to get it:**
  1. Go to https://newsapi.org/register
  2. Sign up
  3. Get key from account page
- **Value:** Your API key
- **Free Tier:** 100 requests/day

#### 🟢 DOCKER REGISTRY

##### For Docker Hub:
- **Name:** `DOCKER_USERNAME`
- **Value:** Your Docker Hub username
- **How to get it:** Your Docker Hub account username

- **Name:** `DOCKER_PASSWORD`
- **Value:** Your Docker Hub password
- **How to get it:** Your Docker Hub account password
- **Security Tip:** Use an access token instead:
  1. Go to https://hub.docker.com/settings/security
  2. Click "New Access Token"
  3. Use token as password

#### 🔵 CLOUD PROVIDER (Choose One)

##### Option A: DigitalOcean
- **Name:** `DIGITALOCEAN_ACCESS_TOKEN`
- **How to get it:**
  1. Go to https://cloud.digitalocean.com/account/api/tokens
  2. Click "Generate New Token"
  3. Name it "GitHub Actions"
  4. Select "Write" scope
  5. Copy the token (you won't see it again!)

- **Name:** `DIGITALOCEAN_CLUSTER_ID`
- **How to get it:**
  1. Create a Kubernetes cluster in DigitalOcean
  2. Go to Kubernetes section
  3. Click on your cluster
  4. Copy the ID from the URL or cluster details

##### Option B: AWS (Alternative)
- **Name:** `AWS_ACCESS_KEY_ID`
- **Name:** `AWS_SECRET_ACCESS_KEY`
- **Name:** `AWS_REGION` (e.g., `us-east-1`)

#### 🟣 OPTIONAL BUT RECOMMENDED

##### SLACK_WEBHOOK_URL
- **Name:** `SLACK_WEBHOOK_URL`
- **How to get it:**
  1. Go to https://api.slack.com/apps
  2. Create new app → From scratch
  3. Add "Incoming Webhooks" feature
  4. Activate and add to workspace
  5. Copy webhook URL
- **Value Example:** `https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXX`

---

## 🛡️ STEP 2: Configure Branch Protection

### 2.1 Navigate to Branch Settings

1. **In Settings** → Click "Branches" in left sidebar
2. Click **"Add branch protection rule"** button

### 2.2 Configure Protection Rules

**Branch name pattern:** `main`

Check these boxes:
- ✅ **Require a pull request before merging**
  - ✅ Require approvals: 1
  - ✅ Dismiss stale pull request approvals
  - ✅ Require review from CODEOWNERS

- ✅ **Require status checks to pass before merging**
  - Search and select: "CI Pipeline"
  - ✅ Require branches to be up to date

- ✅ **Require conversation resolution before merging**

- ✅ **Include administrators** (optional but recommended)

- ✅ **Restrict who can push to matching branches** (optional)
  - Add yourself and team members

Click **"Create"** button at bottom

---

## ⚙️ STEP 3: Enable GitHub Actions

### 3.1 Enable Actions

1. **In Settings** → Click "Actions" → "General"
2. Under "Actions permissions":
   - Select **"Allow all actions and reusable workflows"**

### 3.2 Configure Artifact Settings

Under "Artifact and log retention":
- Set to **7 days** (saves storage costs)

### 3.3 Configure Workflow Permissions

Under "Workflow permissions":
- Select **"Read and write permissions"**
- ✅ **Allow GitHub Actions to create and approve pull requests**

Click **"Save"** at bottom

---

## 🧪 STEP 4: Test Your Setup

### 4.1 Create a Test Branch

```bash
git checkout -b test/cicd-setup
```

### 4.2 Make a Small Change

Create a test file:
```bash
echo "# CI/CD Test" > test_cicd.md
git add test_cicd.md
git commit -m "test: CI/CD pipeline setup"
git push origin test/cicd-setup
```

### 4.3 Create Pull Request

1. Go to your GitHub repository
2. You'll see a yellow banner: "test/cicd-setup had recent pushes"
3. Click **"Compare & pull request"**
4. Add title: "Test CI/CD Pipeline"
5. Click **"Create pull request"**

### 4.4 Watch the CI Pipeline

1. In the PR, scroll down to "Checks"
2. You should see:
   - ⏳ CI Pipeline (running)
   - Various checks running

### 4.5 Expected Results

After 3-5 minutes, you should see:
- ✅ CI Pipeline — All checks have passed
- ✅ backend-quality
- ✅ backend-tests
- ✅ frontend-quality
- ✅ frontend-tests
- ✅ security-scan
- ✅ docker-build

If any fail, check the logs by clicking "Details"

---

## 🔍 STEP 5: Verify Everything Works

### 5.1 Run Local Validation

```bash
# Run the validation script
cd /mnt/c/Users/Devin\ McGrathj/01.project_files/investment_analysis_app
python3 scripts/validate_cicd.py
```

### 5.2 Check GitHub Actions Tab

1. Go to repository → **"Actions"** tab
2. You should see workflows listed:
   - CI Pipeline
   - Security Scan
   - Dependency Updates
   - etc.

### 5.3 Test Staging Deployment

Once your test PR is approved and merged:
1. The staging deployment will trigger automatically
2. Watch in Actions tab → "Staging Deploy" workflow
3. Check for green checkmarks

---

## 🚨 STEP 6: Common Issues & Solutions

### Issue: "Error: secret not found"
**Solution:** Double-check secret name spelling (case-sensitive!)

### Issue: "Docker push failed: unauthorized"
**Solution:** Verify DOCKER_USERNAME and DOCKER_PASSWORD are correct

### Issue: "Database connection failed"
**Solution:** 
1. Check DATABASE_URL format
2. Ensure database is accessible from GitHub Actions
3. May need to whitelist GitHub Actions IP ranges

### Issue: "API rate limit exceeded"
**Solution:** 
1. Check you're using correct API keys
2. Verify free tier limits
3. Implement caching in code

---

## ✅ Final Checklist

### Secrets Added:
- [ ] DATABASE_URL
- [ ] REDIS_URL
- [ ] ALPHA_VANTAGE_API_KEY
- [ ] FINNHUB_API_KEY
- [ ] POLYGON_API_KEY
- [ ] NEWS_API_KEY
- [ ] DOCKER_USERNAME
- [ ] DOCKER_PASSWORD
- [ ] DIGITALOCEAN_ACCESS_TOKEN (or AWS keys)
- [ ] DIGITALOCEAN_CLUSTER_ID (or AWS config)
- [ ] SLACK_WEBHOOK_URL (optional)

### Configuration:
- [ ] Branch protection enabled for main
- [ ] GitHub Actions enabled
- [ ] Artifact retention configured
- [ ] Workflow permissions set

### Testing:
- [ ] Test PR created successfully
- [ ] CI Pipeline passed
- [ ] All checks green
- [ ] Staging deployment works (after merge)

---

## 🎉 Success Indicators

You know your CI/CD is working when:
1. **Every PR** automatically runs tests
2. **Merges to main** trigger staging deployment
3. **No manual steps** required for testing
4. **Slack notifications** arrive (if configured)
5. **Security scans** run daily

---

## 📞 Need Help?

If you encounter issues:

1. **Check the logs**: Click "Details" on failed checks
2. **Review secrets**: Ensure all are added correctly
3. **Test locally**: Run the validation scripts
4. **Check the troubleshooting guide**: TROUBLESHOOTING_GUIDE.md

Remember: The first setup takes time, but once configured, this pipeline will save hours of manual work!

---

## 🚀 Next Steps After Setup

1. **Create your first real feature branch**
2. **Watch the automated testing in action**
3. **Deploy to staging automatically**
4. **Set up production Kubernetes cluster**
5. **Configure monitoring dashboards**

Congratulations! Your Investment Analysis App now has enterprise-grade CI/CD! 🎊