# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import os
import glob
from google import genai
from google.genai import types


def process_pdf(client, pdf_path, model="gemini-2.5-flash-preview-04-17"):
    """Process a single PDF file and return the analysis results."""
    print(f"Processing: {pdf_path}")
    
    try:
        file = client.files.upload(file=pdf_path)
        
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_uri(
                        file_uri=file.uri,
                        mime_type=file.mime_type,
                    ),
                ],
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
            system_instruction=[
                types.Part.from_text(text="""you are a qualified statistician, you will receive a PDF with the findings of an ANOVA analysis that was performed with SPSS. It is your responsibility to comprehend the analysis and provide the findings in text format as shown in the examples. Keep in mind that you should only include group means when the factor is significant.
Remmber that these exaples are for a sample size of 74, and rovided pdfs may have different sample sizes.
Examples:
1) Movement Behaviors (Sitting Transitions)
The result revealed a significant main effect of Information (F(1, 70) = 15.56, p < 0.001, ηp² = 0.18) with sitting transitions significantly higher when information was given (M = 10.69, SE = 1.18) than when it was not given (M = 4.00, SE = 1.22) as shown in Fig. 4. The type of task had no significant effect on sitting transitions (F(1, 70) = 0.67, p = 0.415, ηp² = 0.01). There was no significant interaction between information and task.
 
2) Total Time Spent Sitting
Upon analysis Mauchly's test indicated that the assumption of sphericity had been violated for the main effect of Time (χ²(2) = 12.54, p = 0.002), and consequently for interactions involving Time. Therefore, degrees of freedom were corrected using Greenhouse-Geisser estimates of sphericity (ε = 0.858) for the within-subjects effects. There was a significant main effect of time on total seated duration (F(1.72, 120.05) = 37.25, p < 0.001, ηp² = 0.347) (Fig. 5). Time spent sitting was significantly lower before the earthquake (M = 2.32, SE = 0.80) compared to during (M = 21.18, SE = 1.82; p < .001) and after (M = 10.43, SE = 1.85; p = .001). Time spent sitting during the earthquake was also significantly higher than after earthquake. A significant main effect was found for Task (F(1, 70) = 15.23, p < .001, ηp² = .179) (Fig. 6), with participants in the No Task condition (M = 14.92, SE = 1.36) showing higher overall seated duration than those in the Book Task condition (M = 7.69, SE = 1.26). There was also a significant main effect for information (F(1, 70) = 22.63, p < .001, ηp² = .244) (Fig. 7), with participants given information (M = 15.72, SE = 1.29) having a higher overall seated duration than those not given information (M = 6.90, SE = 1.33). There was a significant interaction between Time and Task (F(1.72, 120.05) = 3.95, p = 0.028, ηp² = .053) (Fig. 8). Before the earthquake, the sitting time was similarly low whether participants were in the Book Task condition (M = 2.26, SE = 1.09) or No Task condition (M = 2.38, SE = 1.18). During the earthquake, sitting time increased substantially more for the No Task group (M = 26.66, SE = 2.67) than the Book Task group (M = 15.69, SE = 2.46). After the earthquake, cover time decreased for both, but remained significantly higher for the No Task group (M = 15.73, SE = 2.73) compared to the Book Task group (M = 5.13, SE = 2.51). There was also a significant interaction between Time and Information (F(1.72, 120.05) = 19.44, p < 0.001, ηp² = 0.217) (Fig. 9). Before the earthquake, sitting duration was low for both the Given (M=1.75, SE=1.12) and Not Given (M=2.89, SE=1.16) information groups. During the earthquake, duration increased drastically  more for the Given group (M=33.37, SE=2.53) compared to the Not Given group (M=8.98, SE=2.61). After the earthquake, duration decreased for both groups but remained higher for the Given group (M=12.02, SE=2.58) than the Not Given group (M=8.83, SE=2.66). The interaction between Task and Information was not significant (F(1, 70) = 0.11, p = 0.747, ηp² = 0.002). Finally, the three-way interaction between time, Task, and information was not significant (F(1.72, 120.05) = 0.29, p = 0.717, ηp² = 0.004).
 
3) Total Time Spent in Cover
Upon analysis Mauchly's test indicated that the assumption of sphericity had been violated for the main effect of Time (χ²(2) = 24.578, p < 0.001), and consequently for interactions involving Time. Therefore, degrees of freedom were corrected using Greenhouse-Geisser estimates of sphericity (ε = 0.769) for the within-subjects effects. There was a significant main effect of time on the duration spent under cover (F(1.54, 107.72) = 12.137, p < 0.001, ηp² = 0.148) (Fig. 10). Post-hoc  comparisons using the Bonferroni correction revealed that time spent under cover was significantly higher during the earthquake (M = 8.09, SE = 1.64) compared to before (M = 0.43, SE = 0.19; p < .001) and after (M = 2.59, SE = 1.09; p = .016). The difference between Before and after was not significant (p = .174). The main effect of Task was not statistically significant (F(1, 70) = 3.593, p = 0.062, ηp² = .049). The main effect of information given was also not significant (F(1, 70) = 1.207, p = 0.276, ηp² = .017). There was a significant interaction between Time and Information (F(1.54, 107.72) = 5.165, p = 0.013, ηp² = 0.069) (Fig. 11). Before the earthquake, cover time was similarly low whether information was given (M = 0.48, SE = 0.26) or not given (M = 0.39, SE = 0.27). During the earthquake, cover time increased substantially for the Given group (M = 11.68, SE = 2.28) than the Not Given group (M = 4.50, SE = 2.36). After the earthquake, cover time decreased for both, but was lower for the Given group (M = 1.17, SE = 1.52) compared to the Not Given group (M = 4.01, SE = 1.57). No significant interaction was found between time and Task (F(1.54, 107.72) = 2.449, p = .105, ηp² = .034). The interaction between Task and information was not significant (F(1, 70) = 0.184, p = .669, ηp² = .003). Finally, the three-way interaction between time, Task, and information was not significant (F(1.54, 107.72) = 0.173, p = .783, ηp² = .002).

4) Number of Items Picked
Upon analysis Mauchly's test indicated that the assumption of sphericity had been violated for the main effect of Time (χ²(2) = 26.691, p < 0.001), and consequently for interactions involving Time. Therefore, degrees of freedom were corrected using Greenhouse-Geisser estimates of sphericity (ε = 0.757) for the within-subjects effects. There was a significant main effect of time on the number of items picked (F(1.51, 106.00) = 40.617, p < 0.001, ηp² = 0.367) (Fig. 12). Post hoc comparisons using the Bonferroni correction revealed that items picked were significantly lower During the earthquake (M = 2.66, SE = 0.32) compared to Before (M = 4.66, SE = 0.30; p < 0.001) and After (M = 7.57, SE = 0.64; p < 0.001). Items picked Before the earthquake were also significantly lower than after the earthquake (p < 0.001). The main effect of Task was not statistically significant (F(1, 70) = 0.265, p = .608, ηp² = .004). The main effect of information given was also not significant (F(1, 70) = 0.002, p = 0.969, ηp² = .000). There was a significant interaction between Time and Task (F(1.51, 106.00) = 5.545, p = 0.010, ηp² = 0.073) (Fig. 13). Before the earthquake, items picked were similar for the Book Task (M = 4.65, SE = 0.41) and No Task (M = 4.67, SE = 0.44) groups. During the earthquake, both groups picked fewer items, but the decrease was numerically larger for the No Task group (M = 2.00, SE = 0.47) than the Book Task group (M = 3.33, SE = 0.43). After the earthquake, both groups increased the number of items picked, but the increase was substantially larger for the No Task group (M = 8.71, SE = 0.94) compared to the Book Task group (M = 6.43, SE = 0.86). No significant interaction was found between Time and Information (F(1.51, 106.00) = 2.648, p = 0.090, ηp² = 0.036). The interaction between Task and Information was not significant (F(1, 70) = 0.489, p = 0.487, ηp² = .007). Finally, the three-way interaction between Time, Task, and Information was not significant (F(1.51, 106.00) = 0.154, p = 0.797, ηp² = 0.002).
 

5) Number of Cover Attempts
Cover attempts were significantly higher when information was given (M = 1.37, SE = 0.17) compared to when it was not given (M = 0.67, SE = 0.18) (F(1, 70) = 7.83, p = 0.007, ηp² = 0.10) (Fig. 14). The type of task had no significant main effect on cover attempts (F(1, 70) = 1.32, p = 0.254, ηp² = 0.02). There was no significant interaction between Task and Information (F(1, 70) = 0.99, p = 0.323, ηp² = 0.01).
"""),
            ],
        )

        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=generate_content_config,
        )
        
        # Get the filename without extension for the report
        filename = os.path.basename(pdf_path)
        
        # Return the analysis with a header for the file
        return f"\n\n{'='*50}\nANALYSIS FOR: {filename}\n{'='*50}\n\n{response.text}"
    
    except Exception as e:
        return f"\n\n{'='*50}\nERROR PROCESSING: {pdf_path}\n{'='*50}\n\nError: {str(e)}"


def generate():
    client = genai.Client(
        api_key="AIzaSyB1qLIlQp2J3EKUIHqMF6rEf0afuOOFFzw",
    )
    
    # Define analysis folder and output file
    analysis_folder = "./Anova analysis"
    output_file = "./Anova analysis/combined_results.txt"
    
    # Find all PDF files in the analysis folder
    pdf_files = glob.glob(os.path.join(analysis_folder, "*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {analysis_folder}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    print(pdf_files)
    # Process each PDF file and collect results
    all_results = []
    for pdf_file in pdf_files:
        result = process_pdf(client, pdf_file)
        all_results.append(result)
        print(f"Completed analysis for {pdf_file}")
    
    # Write all results to a single text file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("STATISTICAL ANALYSIS RESULTS\n")
        f.write(f"Files processed: {len(pdf_files)}\n\n")
        f.write("\n".join(all_results))
    
    print(f"All analyses complete. Results saved to {output_file}")


if __name__ == "__main__":
    generate()
