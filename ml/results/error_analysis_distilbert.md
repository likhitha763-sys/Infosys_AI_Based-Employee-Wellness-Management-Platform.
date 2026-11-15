# Error analysis: distilbert

## SemEval polarity
Errors: 112 of 675 (96 negative tweets predicted positive, 16 positive tweets predicted negative).
Mentions are replaced with @user. 10 random examples of each kind (seed 42).

### Negative predicted as positive
- `2018-En-02742` oh no now that i'm back home i actually have to figure out what i'm gonna do with my life #brilliant 😟
- `2018-En-03724` @user Prime Minister of India & Home Minister of India are responsible to take strict action against BJP 🙄 😄 #sarcasm
- `2018-En-03699` @user Man people wont mourn the 9 chaps beheaded by terrorists. But they will mourn a corrupt killer politician
- `2018-En-00333` My main concern are the children and his wife thats if she is still stuck with him and if she is, then she's strong af. Jeez 😟
- `2018-En-03343` My soon to be ex wife needs to learn to chill out & smile
- `2018-En-02009` Do not linger too long near the howff or you risk the displeasure of a chuhaister with pubes more underwhelming than those of an aurochs.
- `2018-En-01680` The best kind of self harm is through the mind #depression
- `2018-En-00797` What if palindrome were a palindrome? #ironic
- `2018-En-03984` Maybe he was partly right. THESE emails might lead to impeachment and 'lock him up' #ironic #ImpeachTrump
- `2018-En-02911` @user trembled, the #earth quaked, and it became a very great #panic. [2/2]

### Positive predicted as negative
- `2018-En-01352` Do you ever fall asleep before you set your alarm, but then the Lord wakes you up on time anyways 🙌🏻 #WontHeDoIt
- `2018-En-03738` Someone just replied all to an email telling someone else off for sending an email to the whole dept mailing list #ironic 😂
- `2018-En-01926` Worst missing is missing ur ownself u used to be once upon a time. #happy #ambitious#innocent #LifeDreamer
- `2018-En-03192` I'm BEYOND entertained 😂 and in this case it's a sad thing to witness 😂😂 watch and loving the dms 😘 #flattered #cantdoitlikeme #poorthing 😋
- `2018-En-03909` mum got out of a rlly bad car crash completely not injured and i found a rlly sentimental piece of jewellery i thought i'd lost #blessed
- `2018-En-03306` Who also miss glee ? #gleek #glee #GleeForever #DreamsComeTrue #dontstopbelieving #gleecast @user
- `2018-En-03957` Been spoilt even got a Kylie lip kit 😆
- `2018-En-04156` I can't afford to be unhappy. I've made it this far and there isn't anything or anyone that's going to take my happiness away.
- `2018-En-03933` Going to be a student nurse for 4 weeks eeeeppp #newexperience #scared #excitedmuch
- `2018-En-04065` They say money can't buy happiness... Retweet this if you'd be happy with 27 Million Dollars.

## ISEAR: true emotion (rows) against the 8 most common predicted labels (columns)
```
pred              neutral  sadness  fear  realization  anger  joy  disgust  disappointment
emotion_original                                                                          
anger                 318       43     3           34    159    1        6              49
disgust               371       28    11           28     19    2      178              18
fear                  287       62   259           58      8    0        1              18
joy                   307       18     5           80      0  181        0               9
sadness               236      326     4           56      5    3        1              45
```
