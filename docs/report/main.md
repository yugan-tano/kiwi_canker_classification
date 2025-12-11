---
title: 基于SE-ResNet的猕猴桃溃疡病图像分类研究
author: 
  - 张三（组长）
  - 李四（组员A）
  - 王五（组员B）
date: 2025年12月
abstract: |
  本文研究基于SE-ResNet的猕猴桃溃疡病图像分类方法...
geometry: "left=3cm,right=2.5cm,top=2.5cm,bottom=2.5cm"
documentclass: article
classoption: 12pt
numbersections: true
toc: true
toc-depth: 3
header-includes:
  - \usepackage{ctex}
  - \usepackage{graphicx}
  - \usepackage{booktabs}
  - \usepackage{float}
---

\pagenumbering{Roman}
\tableofcontents
\newpage
\listoffigures
\listoftables
\newpage
\pagenumbering{arabic}
\setcounter{page}{1}

\input{01-abstract.md}
\input{02-introduction.md}
\input{03-related-work.md}

\section{方法设计}
\input{04-methodology/01-resnet.md}
\input{04-methodology/02-se-mechanism.md}
\input{04-methodology/03-se-resnet.md}

\section{实验与结果}
\input{05-experiments/01-dataset.md}
\input{05-experiments/02-setup.md}
\input{05-experiments/03-results.md}

\input{06-conclusion.md}

\bibliographystyle{ieeetr}
\bibliography{references}
