# CourseProject

## Documents
Proposal can be found at: [TIS Project Proposal.pdf](https://github.com/QinxiW/TISProject/blob/main/TIS%20Project%20Proposal.pdf)

Progress report can be found at: [TIS Project Progress_Report.pdf](https://github.com/QinxiW/TISProject/blob/main/TIS%20Project%20Progress%20Report.pdf)

Final report can be found at: [TIS Project Documentation.pdf](https://github.com/QinxiW/TISProject/blob/main/TIS%20Project%20Documentation.pdf)

Demo recording can be found at: [TODO], and demo slides can be found at: [TIS Project Documentation Slides](https://github.com/QinxiW/TISProject/blob/main/TIS%20Project%20Presentation.pdf)


## Repo Structure
Final reports contains the details of the project layout, but on a high level:

[/Data](https://github.com/QinxiW/TISProject/tree/main/Data) contains everything exploratory on data analysis, cleanup, and creation

[/Model](https://github.com/QinxiW/TISProject/tree/main/Model) contains everything model experiments, training, and evaluation

[/Search](https://github.com/QinxiW/TISProject/tree/main/Search) contains everything search, retrieval, and model inference along with the artifacts needed

## Quick Start 
Always make sure you are in the parent dir level as the README when you run any of the scripts.
For a quick start, run the following:
```angular2html
python3 -m venv myenv        # I used Python 3.11.4 but compatible with most
source myenv/bin/activate
pip install -r requirements.txt
python Search/search_wine.py 
```

![Run Demo](https://github.com/QinxiW/TISProject/blob/main/ezgif-2-8bdb364349.gif)
