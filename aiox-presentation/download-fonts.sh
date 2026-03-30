# Download Caveat (hand-drawn) and Inter (clean sans-serif) fonts
mkdir -p fonts
# Caveat - hand-drawn style
wget -q "https://fonts.googleapis.com/css2?family=Caveat:wght@400;700&display=swap" -O fonts/caveat.css
# Inter - clean sans-serif
wget -q "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" -O fonts/inter.css
# JetBrains Mono - for terminal
wget -q "https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap" -O fonts/jetbrains.css
echo "Fonts CSS downloaded"
