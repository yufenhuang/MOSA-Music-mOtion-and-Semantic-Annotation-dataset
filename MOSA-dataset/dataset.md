# Dataset Content
MOSA mainly collected violin and piano performaces. The violin performances were performed by the participants in The University of Edinburgh and National Yang-Ming University, and the piano performances were performed by the participants in National Yang-Ming University. Each piece was performed 1-3 trials by the participants. We aligned the recorded audio and 3-D motion data, and several musical infomation were annotated by the experts for each piece. 
The released dataset accordingly includes different types of data:
* **Audio** - `audio.wav`
* **Motion data** and **normalized motion data** - `motion.csv`, `motion_norm.csv`
* **Scores** - `sheet.pdf`, `sheet_annotation.pdf`, `sheet.xml`
* **Annotations** - `(annotaion).csv`
  
The details are described as follows.

## Pieces
We list the violin and piano pieces used in this dataset separately in the following table. Please note that each piece may be split into several sements, and each of them should be seen as an independent performance. Also, some of pieces were performed partly, and we describe them by the number of bars below the Range column.
### Violin
|Code | Name | Segments | Range	|
|--------------|--------------	|--------------	| ----------------------|
|ba1| Bach Violin Partita No.2                              	| 1 | full      | 
|ba3, ba4| Bach Violin Partita No.3                     	    | 2 | full      |
|mo4| Mozart Violin Concerto No.3, mov. 1               	    | 1 | bar 1-53  |
|mo5| Mozart Violin Concerto No.3, mov. 3       	            | 1 | bar 1-188 |
|be4, be5, be6, be7**| Beethoven Violin Sonata No. 5, mov. 1 	| 4 | full      |
|be8| Beethoven Violin Sonata No. 6, mov. 3      	            | 1 | bar 1-64  |
|me4| Mendelssohn Violin Concerto, mov. 1                  	  | 1 | bar 1-103 |
|el1| Elgar Salut d’amour                                 	  | 1 | full      |
|de1| 鄧雨賢(Deng Yu Sian) ⾬夜花                            	| 1 | full      |
|de2| 鄧雨賢(Deng Yu Sian) 望春風      	                      | 1 | full      |

****Note that** the four segments were fully collected by the particpants in National Yang-Ming University only. The data collected in The University of Edinburgh merely contains one segment - be4.

### Piano
|Code | Name | Segments | Range	|
|--------------|--------------	|--------------	| ----------------------|
|ba1| Bach Well Tempered Clavier Prelude No.1       | 1 | full      | 
|ba2| Bach Well Tempered clavier Prelude No.5       | 1 | full      |
|mo1, mo2, mo3| Mozart Piano Sonata No. 11, mov. 1  | 3 | full      |
|mo4| Mozart Piano Sonata No. 11, mov.3     	      | 1 | full      |
|be1| Beethoven Piano Sonata No. 21, mov.1 	        | 1 | bar 1-86  |
|br1, br2| Brahms Intermezzo Op.118, No.2     	    | 2 | full      |
|tc1| Tschaikovsky Four Seasons, Barcarolle, Op.37  | 1 | bar 1-51  |
|ch1| Chopin Grande Valse Brillante Op.18        	  | 1 | bar 1-188 |
|ch2| Chopin Nocturne Op.9, No.2                   	| 1 | full      |
|de1, de2| Debussy Premiere Arabesque L.66 No.1     | 2 | full      |

## Annotations
descriptions
* ``align_notetime.csv``:
* ``beat.csv``:
* ``cadence.csv``:
* ``downbeat.csv``:
* ``expression.csv``:
* ``harmony.csv``:
* ``note.csv``:
* ``phrase_section.csv``:

# Data Formats
```
MOSA_dataset/  
├──instrument/
   ├──piece/
      ├──performer/
         ├──trial/
            ├──audio.wav
            ├──motion.csv
            ├──motion_norm.csv
            ├──annotation/
               ├──sheet.pdf
               ├──sheet_annotation.pdf
               ├──sheet.xml
               ├──annotations/
                  ├──align_notetime.csv
                  ├──...

```
* instrument: there are three folders ``ev``, ``yv`` and ``yp``. ``e`` and ``y`` indicate that the data is collected from The University of Edinburgh or National Yang-Ming University. ``v`` and ``p`` represent violin and piano.
* piece: the code name of each piece. please see the pieces part above. e.g. ``ba1``
* performer: the number of performer which ranged from 01 to 10. e.g. ``ev01``
* trial: the number of trial which ranged from t1 to t3.  

Note that all the files have a prefix ``piece_performer_trial``. For example, the __audio__ of the __first trial__ recorded by the __number one participant__ from __The University of Edinburgh__ with __violin__ piece __Bach Well Tempered Clavier Prelude No.1__ is represnted as ``ba1_ev01_t1_audio.wav``
