# 🎬 Turn Instagram Reels Into Text (Super Easy!)

**Think of this like having a personal assistant that watches Instagram videos and types out everything people say - but it's completely private and runs on your own computer!**

Perfect for:
- 📝 **Content creators** who want to repurpose video content
- 📚 **Students & researchers** studying social media content
- 🎙️ **Podcasters & journalists** extracting quotes from videos
- 🌍 **Anyone** who needs video speech turned into readable text

## ✨ What This App Does (In Simple Terms)

Think of it like this: You know how YouTube can show captions for videos? This does the same thing, but for Instagram Reels, and you get to keep the text!

**Here's what happens:**
1. 📱 You give it a link to an Instagram Reel
2. 🤖 It watches the video and listens carefully
3. ✍️ It types out everything people say in the video
4. 💾 You get the text to save, copy, or use however you want

**The best part?** Everything stays on your computer - nothing gets sent anywhere else on the internet!

---

## 🚀 Super Quick Start (Takes 2 Minutes!)

**Don't worry - this is way easier than it looks! Just follow these 3 simple steps:**

### Step 1: Get the Files 📥
Download this app to your computer (click the green "Code" button above and select "Download ZIP", then unzip it anywhere you like)

### Step 2: Run the Magic Installer ✨
Open the folder you just downloaded and find a file called `install.sh`. Double-click it!

**That's it!** The installer is like a helpful robot that:
- 🤖 Checks if your computer has everything it needs
- 📦 Downloads and installs anything missing (automatically!)
- 🧠 Downloads the AI "brain" that does the transcribing
- ✅ Sets everything up perfectly for you

*This takes about 2 minutes and you don't need to do anything except wait*

### Step 3: Start Using It! 🎉
Look for files called:
- **Windows users**: Double-click `run_app.bat`
- **Mac users**: Double-click `run_app.sh`

**Congratulations!** The app should open and you're ready to turn videos into text!

---

## 📋 What's On This Page

- [⬆️ Super Quick Start](#-super-quick-start-takes-2-minutes) ← **Start here!**
- [🎯 How to Use the App](#-how-to-use-the-app-step-by-step)
- [❓ What if Something Goes Wrong?](#-what-if-something-goes-wrong)
- [🌟 Cool Features You Get](#-cool-features-you-get)
- [🔧 More Detailed Help](#-more-detailed-help-if-needed)

---

## 🎯 How to Use the App (Step by Step)

### The Easy Way (Recommended!)

1. **Double-click the right file for your computer:**
   - Windows: Double-click `run_app.bat`
   - Mac: Double-click `run_app.sh`

2. **A window opens - this is your transcription app!**

3. **Get an Instagram Reel link:**
   - Go to Instagram in your web browser
   - Find a Reel you want to transcribe (it must be public, not private)
   - Click on the Reel to open it
   - Copy the web address from the top of your browser (it looks like: instagram.com/reel/ABC123/)

4. **Paste the link into the app:**
   - Look for a box that says "Instagram Reel URL"
   - Paste your copied link there

5. **Choose your settings (or just leave them as-is):**
   - **Speed vs Quality**: "Base" is perfect for most people
     - Think of it like photo quality: "Tiny" = super fast but okay quality, "Large" = slow but amazing quality
   - **Language**: Usually "Auto-detect" works great

6. **Click "Start Transcription"**
   - The app will show you what it's doing (downloading, listening, typing)
   - This usually takes 1-3 minutes depending on video length

7. **Get your text!**
   - The transcription appears in the big text box
   - Click "Save Transcript" to save it as a text file
   - Or click "Copy to Clipboard" to paste it somewhere else

**That's it!** You just turned a video into text!

## ❓ What if Something Goes Wrong?

**Don't panic!** Here are the most common issues and super simple fixes:

### "I double-clicked install.sh but nothing happened"

**On Mac:**
- Right-click on `install.sh` instead
- Choose "Open With" → "Terminal"
- If it says "permission denied," try this first:
  ```bash
  chmod +x install.sh
  ```

**On Windows:**
- First, make sure you have Python installed from [python.org](https://www.python.org/downloads/)
- Then try double-clicking `install.sh` or open Command Prompt and type:
  ```bash
  bash install.sh
  ```

### "The app won't start when I double-click the run file"

This usually means the installation didn't finish completely. Try this:

1. **Re-run the installer:** Double-click `install.sh` again
2. **Wait for it to completely finish** (you'll see "Installation complete!" or similar)
3. **Then try the run file again**

### "It says the Instagram link is invalid"

- ✅ **Make sure it's a public Reel** (private accounts won't work)
- ✅ **Copy the full web address** from your browser's address bar
- ✅ **The link should look like:** `https://www.instagram.com/reel/ABC123/`

### "The transcription came out blank or gibberish"

- 🎵 **Music-only videos won't have text** (the app needs people talking)
- 🔊 **Very noisy videos might not work well** (background music is usually okay)
- 🗣️ **Try a video where someone is clearly speaking**

### "It's taking forever to download"

- ⏳ **First time use downloads the AI "brain"** (about 150MB - this only happens once!)
- 📶 **Check your internet connection**
- ⚡ **Try using "Tiny" model for faster processing**

### Still Having Trouble?

**Easy solution:** Create a help request [here](https://github.com/christophbeckmann/instagram-reels-transcriber/issues) and tell us:
1. What you were trying to do
2. What error message you saw (if any)
3. Are you using Windows or Mac?

We're friendly and helpful! 🤗

## 🌟 Cool Features You Get

Think of this as your personal video-to-text assistant that:

- 📱 **Works with any public Instagram Reel** - Just copy and paste the link!
- 🎯 **Really accurate transcription** - Uses the same AI technology as professional services
- 🌍 **Understands 50+ languages** - English, Spanish, French, German, Chinese, Arabic, and tons more
- ⚡ **Pretty fast** - Most videos take just 1-3 minutes to transcribe
- 💾 **Lets you save everything** - Keep transcripts as text files or copy to use elsewhere
- 🔒 **Completely private** - Nothing leaves your computer, ever
- 🎨 **Simple to use** - No confusing buttons or technical jargon
- 📊 **Shows you what it's doing** - You can watch the progress so you're never wondering if it's stuck

**Languages it understands:** English, Spanish, French, German, Italian, Portuguese, Dutch, Russian, Chinese, Japanese, Korean, Arabic, Hindi, and many more!

---

## 🔧 More Detailed Help (If Needed)

**Most people won't need this section!** The simple steps above should work for almost everyone. But if you're curious about the details or ran into a more complex issue, here's some extra information.

### Alternative Ways to Start the App

If the double-click method doesn't work for some reason, you can also:

**Method 1: Using Terminal/Command Prompt (Any Computer)**
```bash
# Navigate to your app folder first
cd path/to/your/downloaded/folder

# Then run this:
./install.sh
./run_app.sh  # Mac/Linux
# OR
run_app.bat  # Windows
```

**Method 2: The Technical Way (Advanced Users)**
```bash
# Activate the environment
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows

# Run the app
python main.py
```

### What if You Want to Use the Command Line Version?

Some people prefer typing commands instead of using the visual app. If that's you:

```bash
# First activate the environment (one time)
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows

# Then transcribe a single video
python cli.py https://www.instagram.com/reel/XXXXX/

# Or use interactive mode for multiple videos
python cli.py --interactive
```

### Technical Details About Installation

The `install.sh` script does these things automatically:
- Installs Python 3.12 if you don't have it
- Installs a super-fast package manager called 'uv'
- Creates a safe, isolated space for the app (called a "virtual environment")
- Downloads all the code libraries the app needs
- Downloads the AI model that does the transcription

### If Installation Completely Fails

Try this manual approach:

```bash
# Install uv first (copy the right command for your system)
# Windows (PowerShell):
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Mac/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# Then install the app
uv venv
uv pip sync requirements.txt
```

---

## 📚 Important Information

### Privacy & Safety
This app is **completely private**. Unlike online services, everything happens on your computer:
- ✅ Your videos are never uploaded anywhere
- ✅ The transcripts stay on your computer
- ✅ No accounts or sign-ups required
- ✅ No data tracking or analytics

### Legal & Ethical Use
- 📝 **Only use with public Instagram Reels** (private content isn't accessible anyway)
- 🤝 **Respect content creators** - don't use transcripts to copy or steal content
- 📖 **Follow Instagram's terms of service**
- 💡 **Great for:** research, accessibility, personal notes, content analysis
- ❌ **Not for:** republishing others' content without permission

### What Powers This App
- 🧠 **AI Transcription**: OpenAI's Whisper (the same technology used by professional services)
- 📥 **Video Downloading**: yt-dlp (a powerful, open-source video downloader)
- 🎨 **Interface**: FreeSimpleGUI (simple, clean desktop app framework)
- ⚡ **Fast Installation**: uv package manager (10x faster than traditional Python installers)

---

## 🎓 For Developers & Technical Users

<details>
<summary>Click here for technical details, command line options, and development setup</summary>

### Command Line Options
```bash
# Advanced CLI usage (after activating environment)
python cli.py --model large --language en --verbose https://instagram.com/reel/XXX/
python cli.py --timeout 120 --interactive
```

### Development Setup
```bash
# Install development dependencies
uv pip install -r requirements-dev.txt

# Run tests and linting
uv run pytest
uv run flake8
uv run black .
```

### Managing Dependencies
```bash
# Update dependencies
uv pip sync requirements.txt --upgrade

# Add new packages
uv pip install package-name && uv pip freeze > requirements.txt
```

### Configuration
Edit `config/settings.py` to customize default model size, timeouts, languages, and logging levels.

</details>

---

## 💝 Like This App? Here's How You Can Help!

If this tool saved you time and made your life easier:

- ⭐ **Star this project** - It helps others find it too!
- 🐛 **Report any bugs** - We'll fix them quickly
- 💡 **Suggest improvements** - We love new ideas
- 📣 **Tell your friends** - Share it with anyone who might need it

---

## 🤝 Want to Contribute?

**For developers:** This is an open-source project and we welcome contributions! Check out the technical details section above for development setup.

**For everyone else:** Bug reports and feature suggestions are super valuable too!

---

## 📜 License & Legal

- **Free to use:** MIT License (use it however you want!)
- **Important reminder:** Only use with public content and respect creators' rights
- **Made with ❤️ by:** [Christoph Beckmann](https://github.com/christophbeckmann)

---

**🎉 Enjoy turning those videos into text!**

*Got questions? Found a bug? Want a new feature? [Create an issue here](https://github.com/christophbeckmann/instagram-reels-transcriber/issues) - we're friendly and respond quickly!*
