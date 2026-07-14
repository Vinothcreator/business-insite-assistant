# Mini-project4 Public Deployment Guide

This guide details how to upload your local code to GitHub and deploy it to Render using the pre-configured Docker container (Option B in your walkthrough).

---

## 🛠️ Step 1: Upload Your Code to GitHub

1. Open **[GitHub](https://github.com/)** in your web browser.
2. Sign in and click **New** to create a new repository:
   - **Repository name**: `business-insite-assistant` (or any name you prefer)
   - **Visibility**: Public or Private (according to your preference)
   - **Do NOT check**: "Add a README file", "Add .gitignore", or "Choose a license" (your project already has these configured).
3. Click **Create repository**.
4. In your terminal, run the following commands in the `d:\Data_Analysatics\Mini-project4` directory to link and push your code (replace `<your-username>` with your actual GitHub username):

```bash
# Link your local folder to GitHub
git remote add origin https://github.com/<your-username>/business-insite-assistant.git

# Set your branch name to main
git branch -M main

# Push the code to GitHub
git push -u origin main
```

---

## 🚀 Step 2: Deploy on Render

1. Open your **[Render Dashboard](https://dashboard.render.com/)**.
2. Click **New Web Service ->** (under the "Web Services" card on your dashboard).
3. Connect your GitHub account (if you haven't already), select your `business-insite-assistant` repository, and click **Connect**.
4. On the deployment configuration screen:
   - **Name**: `business-insite-assistant` (or any custom name)
   - **Region**: Select the region closest to you
   - **Branch**: `main`
   - **Runtime**: Select **Docker** (Render should automatically select this because of the `Dockerfile`)
   - **Instance Type**: Select the **Free** tier
5. Click **Create Web Service** at the bottom of the page.

---

## 🔗 Step 3: Access Your Dashboard

1. Render will fetch your code, build the Docker container (which installs the required libraries and MariaDB), and start the services.
2. Once the deploy process says **"Live"**, you will see a public link under the page header (e.g. `https://business-insite-assistant.onrender.com`).
3. Click this link to open your live dashboard in your web browser.

> [!NOTE]
> Since Render's free tier services automatically spin down after 15 minutes of inactivity, your site might take about a minute to boot up the first time you visit it after a long period of inactivity.

---

## 🔑 Option: Adding AI Assistant API Keys
If you want to use the Live AI Assistant modes (GPT-4o-mini / Gemini-2.5-flash) in the cloud:
1. Go to your web service page in the Render dashboard.
2. Click **Environment** on the left menu.
3. Click **Add Environment Variable**:
   - Key: `OPENAI_API_KEY` or `GEMINI_API_KEY`
   - Value: `<your-actual-api-key>`
4. Click **Save Changes**. Render will automatically redeploy the service with access to your keys.
