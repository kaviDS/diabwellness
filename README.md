# Project: diabwellness

Data analysis at Diabwellness based on the clinical data collected at Karunya Sugalaya Healthcare center by Dr. Sivakumar and his team. The data is anonymized by removing confidential information such as Name, Location, Contact, etc. Various information is collected for each patient during each appointment such as prescriptions, vitals, lab results, and measurements. The primary key to combining and indexing this data is called the NFID (a unique patient ID).

## Goals:

There are three major goals that we are working towards in this project which we think can be converted into value-added services. These goals come with their own challenges and difficulties.

1. Medical drug efficiency
2. Risk stratification
3. Prescription prediction

## How to run:

Analysis in the notebooks can be visualized by running these notebooks: `drug_efficiency.ipynb` and `risk_profiling.ipynb` 

## Setup the project environment:

Follow the below steps to setup the project on your local system:

1. Creat an account in "Github".
2. Install "Git" software in your local system using the below link:
        https://git-scm.com/download/win

3. After installing the Git, Clone the repository to your local system :

    Step 1: Open the "Git Bash" and Check the current path by using the command "pwc" (Current working Directory) 
            
             $ pwd
             /c/Users/DELL

    Step 2: Create a folder (eg:"Inten") by using "mkdir" Command. List contents of the current working directory by running "ls" and check whether the created folder "Inten" is displayed in the list. 

             $mkdir Inten
             $ls
    
    Step 3: Change this current working directory, you can use the "cd" command

             $cd Inten

     Now check the current pwd of inten folder:
            
            $ pwd
            /c/Users/DELL/Inten

    Step 4: Once you create a folder you can clone the repo to that "Inten" folder by using the Command "git clone".
               => First go to github and copy the repo path that you want to clone 
               => Then past the path in this command.
    
            $ git clone https://github.com/diabwellness-ai/optimizerx-cdss.git

4. Create the conda environment:

    Step 1: Install the Anaconda framework by using below link and Open the Anaconda Powershell Prompt.

            https://docs.anaconda.com/anaconda/install/windows/

    Step 2: Run the command "conda env create -f " and give the path of the file "environment.yml".

            (base) PS C:\Users\DELL> conda env create -f C:\Users\DELL\Inten\diabwellness\environment.yml 

    Step 3: To install the diabwellness package run "python setup.py install".
            
            (base) PS C:\WINDOWS\system 32> python C:\Users\DELL\Inten\diabwellness\setup.py install
 
    Step 4: Now open Jupyter Notebook and run some notebook file from the Intern folder and you can see some output.

Above listed steps and commands will help to setup the project environment.