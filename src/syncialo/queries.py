"""LMQL queries (deprecated)"""

import lmql


ARGUMENT_INFO = """A crucial part of critical thinking is to identify, construct, and evaluate arguments.

In everyday life, people often use "argument" to mean a quarrel between people. But in logic and critical thinking, an argument is a list of statements, one of which is the conclusion and the others are the premises or assumptions of the argument.

To give an argument is to provide a set of premises as reasons for accepting the conclusion. To give an argument is not necessarily to attack or criticize someone. Arguments can also be used to support other people's viewpoints.

Here is an example of an argument:

> If you want to find a good job, you should work hard. You do want to find a good job. So you should work hard.

The first two sentences here are the premises of the argument, and the last sentence is the conclusion. To give this argument is to offer the premises as reasons for accepting the conclusion."""

# keywords
PRO = "PRO"
CON = "CON"

@lmql.query
async def suggest_topics(tags: list, debates_per_tag_cluster: int):
    '''lmql
    """
    ### User:
    Task: Suggest interesting and controversial debating topics.

    Our group is planning a debate and we're searching for novel suitable topics.
    
    The debate is supposed to be explicitly related to the following issues:

    {' - '.join(tags)}\n

    Each topic you suggest should be tailored to these issues and specifically reflect at least two of the issues (but more is better).

    Can you please state {debates_per_tag_cluster} different debating topics which touch upon the above issues and from which we choose the most suitable ones? Be creative!\n
    
    ### Assistant:
    
    Yes, for sure. You may pick one of the following topics:\n
    """
    topics = []
    for i in range(10):
        "{i+1}. [TOPIC]" where STOPS_AT(TOPIC, "\n")
        topic = TOPIC.strip("\n ")
        if not topic:
            break
        topics.append(topic)
        
    return topics
    '''

    
@lmql.query
async def suggest_motion(topic: str, tags: list):
    '''lmql
    """
    ### User:
    Task: Suggest a suitable motion for our debate.

    Our group is planning a debate about the topic:\n
    
    {topic}\n
    
    The overarching issues of the debates are: {' - '.join(tags)}\n

    Can you please state a precise and very concise motion, or central claim for our debate? Hints:
    
    - The motion should take a clear and unequivocal stance.
    - The motion expresses the view of the 'pro'-side in the debate.
    - The motion appeals to persons concerned about the overarching issues listed above. 
    - DON'T start with \"This house ...\".
    - DON'T start with \"This debate ...\".\n
    
    ### Assistant:
    
    Yes, for sure. You may start your debate from the following central claim:\n
    """
    "[CLAIM]" where STOPS_AT(CLAIM, "\n")
    
    
    claim = CLAIM.strip("\n\"\' ")
    if not claim.startswith("This house") and not claim.startswith("This debate"):
        return claim

    """
    ### User:
    
    I've noticed that the claim starts with 'This house ...' / 'This debate ...'
    
    Can you please simplify and reformulate the claim by dropping this phrase?\n

    ### Assistant:
    
    I'm sorry, here's the revised claim:\n
    """
    "[CLAIM]" where STOPS_AT(CLAIM, "\n")
        
    return CLAIM.strip("\n\"\' ")
    '''


@lmql.query
async def identify_premises(argument: str, conclusion: str, valence: str):
    '''lmql
    valence_text = "in support of" if valence == PRO else "against"
    """
    ### User:
    Task: Identify premises of an argument.

    Read the following background information carefully before answering!

    /// background_information
    {ARGUMENT_INFO}
    ///

    Now, you've previously maintained in a debate that:

    [[A]] {argument}

    which you've advanced as a reason {valence_text}:

    [[B]] {conclusion}

    Can you please identify the premises (up to 5) of the argument [[A]]? State each premise as a single, concise sentence.\n
    
    ### Assistant:
    
    Yes, for sure. The premise(s) of the argument [[A]] are:\n
    """
    premises = []
    "  Premise 1."
    while True:
        " [PREMISE]" where STOPS_AT(PREMISE, "\n")
        premise = PREMISE.strip("\n ")
        if not premise:
            break
        premises.append(premise)
        i = len(premises)
        " [LABEL]" where LABEL in [f" Premise {i+1}.", "\n"]
        if not LABEL.strip("\n ") or i>=4:
            break
    return premises
    '''

        

@lmql.query
async def rank_by_plausibility(premises: list, tags:list = []):
    '''lmql
    labels = [f"P{i+1}" for i, _ in enumerate(premises)]
    """
    ### User:
    Task: Rank the premises in an argument according to plausibility
    
    Domain: {' - '.join(tags)}

    Read the following background information carefully before answering!

    /// background_information
    {ARGUMENT_INFO}
    ///

    Now, your opponent has previously maintained in a debate that:\n

    """
    for i, premise in enumerate(premises):
        "P{i+1}. {premise}\n"
    """
    Which of these premises are presumably the most plausible ones (i.e., the strongest, the most convincing ones)? Can you order the premises (labels) from very plausible to very unplausible?\n

    ### Assistant:
    
    Yes, for sure. This is a ranking of the premises (beginning with the most plausible one):
    """
    selected = []
    remaining = labels
    for i in range(len(premises)):
        "\n{i+1}. [LABEL]" where LABEL in remaining
        selected.append(LABEL)
        remaining.remove(LABEL)
        
    return [int(l[1:])-1 for l in selected]
    '''

    
@lmql.query
async def supporting_argument(premises: list, target_idx: int, n:int = 3, tags:list = []):
    '''lmql
    nth = ['first','second','third','fourth','fifth'][target_idx] if target_idx<5 else f"{target_idx}th"
    """
    ### User:
    Task: Provide additional supporting arguments for a given claim
    
    Domain tags: {' - '.join(tags)}

    Read the following background information carefully before answering!

    /// background_information
    {ARGUMENT_INFO}
    ///

    Now, you've previously maintained in a debate that:\n

    """
    for i, premise in enumerate(premises):
        "P{i+1}. {premise}\n"
    """
    Can you provide up to {n} different and independent arguments -- each consisting in a catchy title and a single concise statement, formatted as '**title:** statement'  -- that back up the {nth} proposition? Make sure your arguments argue for proposition P{target_idx+1} in specific and plausible ways without merely repeating that proposition. Be inspired by the domain tags above.\n
    
    ### Assistant:
    
    Yes, for sure. Arguments in support of the claim '{premises[target_idx]}' are:\n
    """
    arguments = []
    for i in range(n):
        "{i+1}. [ARGUMENT]" where STOPS_AT(ARGUMENT, "\n")
        argument = ARGUMENT.strip("\n ")
        if not argument:
            break
        argument = argument.replace("**", "")
        if "Title: " in argument:
            argument = argument.replace("Title: ", "")
        arguments.append(argument)

    return arguments
    '''

    

@lmql.query
async def attacking_argument(premises: list, target_idx: int, n:int = 3, tags:list = []):
    '''lmql
    nth = ['first','second','third','fourth','fifth'][target_idx]
    """
    ### User:
    Task: Provide objections against a given claim
    
    Domain: {' - '.join(tags)}

    Read the following background information carefully before answering!

    /// background_information
    {ARGUMENT_INFO}
    ///

    Now, your opponent has previously maintained in a debate that:\n

    """
    for i, premise in enumerate(premises):
        "P{i+1}. {premise}\n"
    """
    Can you provide up to {n} diverse arguments -- each consisting in a catchy title and a single concise statement, formatted as '**title:** statement' -- that object to the {nth} claim? Make sure your arguments contain specific and plausible considerations that argue why proposition P{target_idx+1} is false. Be inspired by the domain tags above.\n
    
    ### Assistant:
    
    Yes, for sure. Objections against the claim '{premises[target_idx]}' are:\n
    """
    arguments = []
    for i in range(n):
        "{i+1}. [ARGUMENT]" where STOPS_AT(ARGUMENT, "\n")
        argument = ARGUMENT.strip("\n ")
        if not argument:
            break
        argument = argument.replace("**", "")
        if "Title: " in argument:
            argument = argument.replace("Title: ", "")
        arguments.append(argument)
        
    return arguments
    '''
    