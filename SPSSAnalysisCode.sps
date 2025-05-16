* Encoding: UTF-8.
* Encoding: .
GLM ItemsPickedBeforeEarthquake ItemsPickedDuringEarthquake ItemsPickedAfterEarthquake BY Task 
    Information 
  /WSFACTOR=Time 3 Polynomial 
  /METHOD=SSTYPE(3) 
  /POSTHOC=Task Information(TUKEY LSD QREGW) 
  /PLOT=PROFILE(Task Information Time Time*Information Time*Task Time*Information*Task) TYPE=LINE 
    ERRORBAR=NO MEANREFERENCE=NO YAXIS=AUTO 
  /EMMEANS=TABLES(OVERALL) 
  /EMMEANS=TABLES(Task) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Information) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Time) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Task*Information) 
  /EMMEANS=TABLES(Task*Time) 
  /EMMEANS=TABLES(Information*Time) 
  /EMMEANS=TABLES(Task*Information*Time) 
  /PRINT=DESCRIPTIVE ETASQ OPOWER HOMOGENEITY 
  /CRITERIA=ALPHA(.05) 
  /WSDESIGN=Time 
  /DESIGN=Task Information Task*Information.

OUTPUT EXPORT
  /CONTENTS EXPORT=VISIBLE
  /PDF DOCUMENTFILE= 'D:\Acads\mtp\Data_Analsysis\Anova analysis\Items_picked_Task_Information.pdf'.

OUTPUT CLOSE ALL.

GLM TotalDurationIntableCoverBeforeEarthquake TotalDurationIntableCoverDuringEarthquake TotalDurationIntableCoverAfterEarthquake BY Task 
    Information 
  /WSFACTOR=Time 3 Polynomial 
  /METHOD=SSTYPE(3) 
  /POSTHOC=Task Information(TUKEY LSD QREGW) 
  /PLOT=PROFILE(Task Information Time Time*Information Time*Task Time*Information*Task) TYPE=LINE 
    ERRORBAR=NO MEANREFERENCE=NO YAXIS=AUTO 
  /EMMEANS=TABLES(OVERALL) 
  /EMMEANS=TABLES(Task) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Information) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Time) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Task*Information) 
  /EMMEANS=TABLES(Task*Time) 
  /EMMEANS=TABLES(Information*Time) 
  /EMMEANS=TABLES(Task*Information*Time) 
  /PRINT=DESCRIPTIVE ETASQ OPOWER HOMOGENEITY 
  /CRITERIA=ALPHA(.05) 
  /WSDESIGN=Time 
  /DESIGN=Task Information Task*Information.

OUTPUT EXPORT
  /CONTENTS EXPORT=VISIBLE
  /PDF DOCUMENTFILE= 'D:\Acads\mtp\Data_Analsysis\Anova analysis\total_duration_in_table_cover_across_time.pdf'.

OUTPUT CLOSE ALL.

GLM BooksPlacedBeforeEarthquake BooksPlacedDuringEarthquake BooksPlacedAfterEarthquake BY Information 
  /WSFACTOR=Time 3 Polynomial 
  /METHOD=SSTYPE(3) 
  /POSTHOC= Information(TUKEY LSD QREGW) 
  /PLOT=PROFILE(Information Time Time*Information) TYPE=LINE 
    ERRORBAR=NO MEANREFERENCE=NO YAXIS=AUTO 
  /EMMEANS=TABLES(OVERALL) 
  /EMMEANS=TABLES(Information) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Time) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Information*Time) 
  /PRINT=DESCRIPTIVE ETASQ OPOWER HOMOGENEITY 
  /CRITERIA=ALPHA(.05) 
  /WSDESIGN=Time 
  /DESIGN=Information.

OUTPUT EXPORT
  /CONTENTS EXPORT=VISIBLE
  /PDF DOCUMENTFILE= 'D:\Acads\mtp\Data_Analsysis\Anova analysis\books_placed_across_time.pdf'.

OUTPUT CLOSE ALL.
  
GLM ItemsObservedBeforeEarthquake ItemsObservedDuringEarthquake ItemsObservedAfterEarthquake BY Information 
  /WSFACTOR=Time 3 Polynomial 
  /METHOD=SSTYPE(3) 
  /POSTHOC= Information(TUKEY LSD QREGW) 
  /PLOT=PROFILE(Information Time Time*Information) TYPE=LINE 
    ERRORBAR=NO MEANREFERENCE=NO YAXIS=AUTO 
  /EMMEANS=TABLES(OVERALL) 
  /EMMEANS=TABLES(Information) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Time) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Information*Time) 
  /PRINT=DESCRIPTIVE ETASQ OPOWER HOMOGENEITY 
  /CRITERIA=ALPHA(.05) 
  /WSDESIGN=Time 
  /DESIGN=Information.

OUTPUT EXPORT
  /CONTENTS EXPORT=VISIBLE
  /PDF DOCUMENTFILE= 'D:\Acads\mtp\Data_Analsysis\Anova analysis\items_observed_across_time.pdf'.

OUTPUT CLOSE ALL.

UNIANOVA CoverAttempts BY Information Task 
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /POSTHOC=Information Task(TUKEY LSD BONFERRONI QREGW) 
  /PLOT=PROFILE(Information Task Information*Task) TYPE=LINE ERRORBAR=NO MEANREFERENCE=NO YAXIS=AUTO 
  /EMMEANS=TABLES(OVERALL) 
  /EMMEANS=TABLES(Information) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Task) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Information*Task) 
  /PRINT ETASQ DESCRIPTIVE HOMOGENEITY OPOWER 
  /CRITERIA=ALPHA(.05) 
  /DESIGN=Information Task Information*Task.

OUTPUT EXPORT
  /CONTENTS EXPORT=VISIBLE
  /PDF DOCUMENTFILE= 'D:\Acads\mtp\Data_Analsysis\Anova analysis\cover_attempts.pdf'.

OUTPUT CLOSE ALL.

UNIANOVA AverageDurationInTableCover BY Information Task 
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /POSTHOC=Information Task(TUKEY LSD BONFERRONI QREGW) 
  /PLOT=PROFILE(Information Task Information*Task) TYPE=LINE ERRORBAR=NO MEANREFERENCE=NO YAXIS=AUTO 
  /EMMEANS=TABLES(OVERALL) 
  /EMMEANS=TABLES(Information) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Task) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Information*Task) 
  /PRINT ETASQ DESCRIPTIVE HOMOGENEITY OPOWER 
  /CRITERIA=ALPHA(.05) 
  /DESIGN=Information Task Information*Task.

OUTPUT EXPORT
  /CONTENTS EXPORT=VISIBLE
  /PDF DOCUMENTFILE= 'D:\Acads\mtp\Data_Analsysis\Anova analysis\average_duration_in_table_cover.pdf'.

OUTPUT CLOSE ALL.

UNIANOVA TotalDurationIntableCover BY Information Task 
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /POSTHOC=Information Task(TUKEY LSD BONFERRONI QREGW) 
  /PLOT=PROFILE(Information Task Information*Task) TYPE=LINE ERRORBAR=NO MEANREFERENCE=NO YAXIS=AUTO 
  /EMMEANS=TABLES(OVERALL) 
  /EMMEANS=TABLES(Information) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Task) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Information*Task) 
  /PRINT ETASQ DESCRIPTIVE HOMOGENEITY OPOWER 
  /CRITERIA=ALPHA(.05) 
  /DESIGN=Information Task Information*Task.

OUTPUT EXPORT
  /CONTENTS EXPORT=VISIBLE
  /PDF DOCUMENTFILE= 'D:\Acads\mtp\Data_Analsysis\Anova analysis\total_duration_in_table_cover.pdf'.

OUTPUT CLOSE ALL.

UNIANOVA SittingTransitions BY Information Task 
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /POSTHOC=Information Task(TUKEY LSD BONFERRONI QREGW) 
  /PLOT=PROFILE(Information Task Information*Task) TYPE=LINE ERRORBAR=NO MEANREFERENCE=NO YAXIS=AUTO 
  /EMMEANS=TABLES(OVERALL) 
  /EMMEANS=TABLES(Information) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Task) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Information*Task) 
  /PRINT ETASQ DESCRIPTIVE HOMOGENEITY OPOWER 
  /CRITERIA=ALPHA(.05) 
  /DESIGN=Information Task Information*Task.

OUTPUT EXPORT
  /CONTENTS EXPORT=VISIBLE
  /PDF DOCUMENTFILE= 'D:\Acads\mtp\Data_Analsysis\Anova analysis\sitting_transitions.pdf'.

OUTPUT CLOSE ALL.

UNIANOVA AverageSeatedDuration BY Information Task 
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /POSTHOC=Information Task(TUKEY LSD BONFERRONI QREGW) 
  /PLOT=PROFILE(Information Task Information*Task) TYPE=LINE ERRORBAR=NO MEANREFERENCE=NO YAXIS=AUTO 
  /EMMEANS=TABLES(OVERALL) 
  /EMMEANS=TABLES(Information) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Task) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Information*Task) 
  /PRINT ETASQ DESCRIPTIVE HOMOGENEITY OPOWER 
  /CRITERIA=ALPHA(.05) 
  /DESIGN=Information Task Information*Task.

OUTPUT EXPORT
  /CONTENTS EXPORT=VISIBLE
  /PDF DOCUMENTFILE= 'D:\Acads\mtp\Data_Analsysis\Anova analysis\average_seated_duration.pdf'.

OUTPUT CLOSE ALL.

UNIANOVA TotalSeatedDuration BY Information Task 
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /POSTHOC=Information Task(TUKEY LSD BONFERRONI QREGW) 
  /PLOT=PROFILE(Information Task Information*Task) TYPE=LINE ERRORBAR=NO MEANREFERENCE=NO YAXIS=AUTO 
  /EMMEANS=TABLES(OVERALL) 
  /EMMEANS=TABLES(Information) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Task) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Information*Task) 
  /PRINT ETASQ DESCRIPTIVE HOMOGENEITY OPOWER 
  /CRITERIA=ALPHA(.05) 
  /DESIGN=Information Task Information*Task.

OUTPUT EXPORT
  /CONTENTS EXPORT=VISIBLE
  /PDF DOCUMENTFILE= 'D:\Acads\mtp\Data_Analsysis\Anova analysis\total_seated_duration.pdf'.

OUTPUT CLOSE ALL.

GLM TotalSeatedDurationBeforeEarthquake TotalSeatedDurationDuringEarthquake TotalSeatedDurationAfterEarthquake BY Task 
    Information 
  /WSFACTOR=Time 3 Polynomial 
  /METHOD=SSTYPE(3) 
  /POSTHOC=Task Information(TUKEY LSD QREGW) 
  /PLOT=PROFILE(Task Information Time Time*Information Time*Task Time*Information*Task) TYPE=LINE 
    ERRORBAR=NO MEANREFERENCE=NO YAXIS=AUTO 
  /EMMEANS=TABLES(OVERALL) 
  /EMMEANS=TABLES(Task) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Information) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Time) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Task*Information) 
  /EMMEANS=TABLES(Task*Time) 
  /EMMEANS=TABLES(Information*Time) 
  /EMMEANS=TABLES(Task*Information*Time) 
  /PRINT=DESCRIPTIVE ETASQ OPOWER HOMOGENEITY 
  /CRITERIA=ALPHA(.05) 
  /WSDESIGN=Time 
  /DESIGN=Task Information Task*Information.

OUTPUT EXPORT
  /CONTENTS EXPORT=VISIBLE
  /PDF DOCUMENTFILE= 'D:\Acads\mtp\Data_Analsysis\Anova analysis\total_seated_duration_across_time.pdf'.

OUTPUT CLOSE ALL.

GLM TimeNearCornerBeforeEarthquake TimeNearCornerDuringEarthquake TimeNearCornerAfterEarthquake BY Task 
    Information 
  /WSFACTOR=Time 3 Polynomial 
  /METHOD=SSTYPE(3) 
  /POSTHOC=Task Information(TUKEY LSD QREGW) 
  /PLOT=PROFILE(Task Information Time Time*Information Time*Task Time*Information*Task) TYPE=LINE 
    ERRORBAR=NO MEANREFERENCE=NO YAXIS=AUTO 
  /EMMEANS=TABLES(OVERALL) 
  /EMMEANS=TABLES(Task) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Information) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Time) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Task*Information) 
  /EMMEANS=TABLES(Task*Time) 
  /EMMEANS=TABLES(Information*Time) 
  /EMMEANS=TABLES(Task*Information*Time) 
  /PRINT=DESCRIPTIVE ETASQ OPOWER HOMOGENEITY 
  /CRITERIA=ALPHA(.05) 
  /WSDESIGN=Time 
  /DESIGN=Task Information Task*Information.

OUTPUT EXPORT
  /CONTENTS EXPORT=VISIBLE
  /PDF DOCUMENTFILE='D:\Acads\mtp\Data_Analsysis\Anova analysis\time_near_corner_across_time.pdf'.

OUTPUT CLOSE ALL.

GLM TimeNearWallBeforeEarthquake TimeNearWallDuringEarthquake TimeNearWallAfterEarthquake BY Task 
    Information 
  /WSFACTOR=Time 3 Polynomial 
  /METHOD=SSTYPE(3) 
  /POSTHOC=Task Information(TUKEY LSD QREGW) 
  /PLOT=PROFILE(Task Information Time Time*Information Time*Task Time*Information*Task) TYPE=LINE 
    ERRORBAR=NO MEANREFERENCE=NO YAXIS=AUTO 
  /EMMEANS=TABLES(OVERALL) 
  /EMMEANS=TABLES(Task) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Information) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Time) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Task*Information) 
  /EMMEANS=TABLES(Task*Time) 
  /EMMEANS=TABLES(Information*Time) 
  /EMMEANS=TABLES(Task*Information*Time) 
  /PRINT=DESCRIPTIVE ETASQ OPOWER HOMOGENEITY 
  /CRITERIA=ALPHA(.05) 
  /WSDESIGN=Time 
  /DESIGN=Task Information Task*Information.

OUTPUT EXPORT
  /CONTENTS EXPORT=VISIBLE
  /PDF DOCUMENTFILE= 'D:\Acads\mtp\Data_Analsysis\Anova analysis\time_near_wall_across_time.pdf'.

OUTPUT CLOSE ALL.

UNIANOVA TotalTimeNearCorner BY Information Task 
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /POSTHOC=Information Task(TUKEY LSD BONFERRONI QREGW) 
  /PLOT=PROFILE(Information Task Information*Task) TYPE=LINE ERRORBAR=NO MEANREFERENCE=NO YAXIS=AUTO 
  /EMMEANS=TABLES(OVERALL) 
  /EMMEANS=TABLES(Information) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Task) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Information*Task) 
  /PRINT ETASQ DESCRIPTIVE HOMOGENEITY OPOWER 
  /CRITERIA=ALPHA(.05) 
  /DESIGN=Information Task Information*Task.

OUTPUT EXPORT
  /CONTENTS EXPORT=VISIBLE
  /PDF DOCUMENTFILE= 'D:\Acads\mtp\Data_Analsysis\Anova analysis\total_time_near_corner.pdf'.

OUTPUT CLOSE ALL.

UNIANOVA TotalTimeNearWall BY Information Task 
  /METHOD=SSTYPE(3) 
  /INTERCEPT=INCLUDE 
  /POSTHOC=Information Task(TUKEY LSD BONFERRONI QREGW) 
  /PLOT=PROFILE(Information Task Information*Task) TYPE=LINE ERRORBAR=NO MEANREFERENCE=NO YAXIS=AUTO 
  /EMMEANS=TABLES(OVERALL) 
  /EMMEANS=TABLES(Information) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Task) COMPARE ADJ(BONFERRONI) 
  /EMMEANS=TABLES(Information*Task) 
  /PRINT ETASQ DESCRIPTIVE HOMOGENEITY OPOWER 
  /CRITERIA=ALPHA(.05) 
  /DESIGN=Information Task Information*Task.

OUTPUT EXPORT
  /CONTENTS EXPORT=VISIBLE
  /PDF DOCUMENTFILE= 'D:\Acads\mtp\Data_Analsysis\Anova analysis\total_time_near_wall.pdf'.

OUTPUT CLOSE ALL.
