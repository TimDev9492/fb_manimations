file="$1"
scene="$2"

presentation_dir="./presentations"

# Replace the import
sed -i 's/^from manim\.manimlib import \*$/from manimlib import \*/' "$file"

# Render the scene
manim-slides render --GL "$file" "$scene"

# Revert import replacement
sed -i 's/^from manimlib import \*$/from manim\.manimlib import \*/' "$file"

# Convert to html presentation
html_file="${presentation_dir}/${scene}.html"
manim-slides convert --one-file "$scene" "$html_file"

# Open presentation in browser
# brave "$html_file"
