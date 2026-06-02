import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class MiningDataPipeline:
    def __init__(self, file_path):
        """
        Initializes the pipeline with the target dataset path.
        """
        self.raw_path = file_path
        self.df = None
        
    def ingest_data(self):
        """
        Stage 1: Data Ingestion
        Loads the raw CSV dataset into a Pandas DataFrame with robust error handling.
        """
        print("Executing Stage 1: Ingesting raw telemetry data...")
        try:
            self.df = pd.read_csv(self.raw_path)
            print(f"Success! Dataset loaded completely.")
            print(f"   Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns\n")
            
            print("Detected Attributes in Telemetry Stream:")
            print(list(self.df.columns))
            
        except FileNotFoundError:
            print(f"Error: Could not find the dataset at '{self.raw_path}'.")
            print("   Please check that your file is named correctly and located in the data/ folder.")
        except pd.errors.EmptyDataError:
            print("Error: The dataset file is completely empty.")
        except Exception as e:
            print(f"An unexpected error occurred during ingestion: {e}")

    def clean_data(self, month_filter="2017-06"):
        """
        Stage 2: Data Cleaning & Automated Filtering
        Removes duplicates, handles nulls, fixes decimal formatting, and slices a unique window.
        """
        print("Executing Stage 2: Automated cleaning and unique programmatic filtering...")
        try:
            if self.df is None:
                print("Error: No data available to clean. Run ingest_data() first.")
                return

            # 1. Deduplication
            initial_rows = self.df.shape[0]
            self.df.drop_duplicates(inplace=True)
            dropped_dupes = initial_rows - self.df.shape[0]
            print(f"   Removed {dropped_dupes} duplicate records.")

            # 2. Missing Value Imputation
            null_counts = self.df.isnull().sum().sum()
            if null_counts > 0:
                self.df.ffill(inplace=True)
                print(f"   Handled {null_counts} missing values using forward-fill.")
            else:
                print("   Data Integrity Verified: 0 missing values detected.")

            # 3. European Decimal Conversion Fix (Strings with commas -> Mathematical floats)
            columns_to_fix = ['% Iron Feed', '% Silica Feed', '% Silica Concentrate']
            for col in columns_to_fix:
                if col in self.df.columns:
                    self.df[col] = self.df[col].astype(str).str.replace(',', '.')
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
            self.df.ffill(inplace=True)
            print("   DataType Correction: Transformed text-based decimals into math-ready float types.")

            # 4. Data Type Correction for Date & Slicing Unique Window
            self.df['date'] = pd.to_datetime(self.df['date'])
            print(f"   Applying unique programmatic filter for production window: {month_filter}")
            self.df = self.df[self.df['date'].dt.to_period('M') == month_filter]
            
            print(f"   Cleaning complete! Filtered Dataset Shape: {self.df.shape[0]} rows, {self.df.shape[1]} columns\n")
            
        except Exception as e:
            print(f"Error: An error occurred during data processing/cleaning: {e}")

    def analyze_statistics(self):
        """
        Stage 3: Engineering Data Analytics
        Mandatory implementation of NumPy for calculating descriptive and distribution metrics.
        """
        print("Executing Stage 3: Statistical computation via NumPy...")
        try:
            if self.df is None:
                print("Error: No data available for analysis. Run clean_data() first.")
                return

            # Convert key engineering columns into raw NumPy arrays
            iron_feed = self.df['% Iron Feed'].to_numpy()
            silica_feed = self.df['% Silica Feed'].to_numpy()
            silica_concentrate = self.df['% Silica Concentrate'].to_numpy()

            print("\n=======================================================")
            print("       NUMPY METALLURGICAL PRODUCTION STATISTICS       ")
            print("=======================================================")
            
            for name, array in [("Raw Iron Feed (%)", iron_feed), 
                                ("Raw Silica Feed (%)", silica_feed), 
                                ("Final Silica Concentrate (%)", silica_concentrate)]:
                
                mean_val = np.mean(array)
                median_val = np.median(array)
                std_val = np.std(array)
                var_val = np.var(array)
                
                # Manual skewness calculation using NumPy
                n = len(array)
                mean_diff = array - mean_val
                skew_val = (np.sum(mean_diff ** 3) / n) / (std_val ** 3)

                print(f"\nMetrics for {name}:")
                print(f"   * Mean (Average)     : {mean_val:.4f}")
                print(f"   * Median (Middle)    : {median_val:.4f}")
                print(f"   * Standard Deviation : {std_val:.4f}")
                print(f"   * Variance (Spread)  : {var_val:.4f}")
                print(f"   * Distribution Skew  : {skew_val:.4f}")

            # >>> HERE IS WHERE IT GOES <<<
            # Correlation Matrix Analysis via NumPy
            print("\nProcess Telemetry Correlation Analysis:")
            matrix_data = np.vstack([iron_feed, silica_feed, silica_concentrate])
            corr_matrix = np.corrcoef(matrix_data)
            
            print(f"   * Feed Iron vs. Feed Silica Correlation : {corr_matrix[0, 1]:.4f}")
            print(f"   * Feed Silica vs. Final Slag Impurity   : {corr_matrix[1, 2]:.4f}")
            print("=======================================================\n")

        except KeyError as e:
            print(f"Error: Column Naming Error: Could not find the column {e} in your dataset.")
        except Exception as e:
            print(f"Error: An error occurred during statistical analytics: {e}")

    def generate_visualizations(self):
        """
        Stage 4: Data Visualization & Animation
        Generates 3 separate static graphs and 2 separate animated graphs with clean text formatting.
        """
        print("Executing Stage 4: Rendering and saving separate graphs...")
        try:
            if self.df is None:
                print("Error: No data available to plot. Run clean_data() first.")
                return

            iron_feed = self.df['% Iron Feed'].to_numpy()
            silica_feed = self.df['% Silica Feed'].to_numpy()
            silica_concentrate = self.df['% Silica Concentrate'].to_numpy()
            dates = self.df['date'].to_numpy()

            # --- STATIC PLOTS ---
            print("   Generating separate static plots...")
            
            # 1. Histogram
            plt.figure(figsize=(6, 4.5))
            plt.hist(silica_concentrate, bins=30, color='salmon', edgecolor='black', alpha=0.7)
            plt.title('Distribution of Final Silica Concentrate')
            plt.xlabel('% Silica Concentrate')
            plt.ylabel('Frequency')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig('static_histogram.png', dpi=150)
            plt.close()

            # 2. Boxplot
            plt.figure(figsize=(6, 4.5))
            plt.boxplot([iron_feed, silica_feed], labels=['% Iron Feed', '% Silica Feed'], patch_artist=True,
                        boxprops=dict(facecolor='lightblue', color='blue'))
            plt.title('Process Feed Quality Spread & Outliers')
            plt.ylabel('Percentage (%)')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig('static_boxplot.png', dpi=150)
            plt.close()

            # 3. Scatter Plot
            plt.figure(figsize=(6, 4.5))
            plt.scatter(silica_feed, silica_concentrate, alpha=0.3, color='teal', s=1)
            plt.title('Feed Silica vs. Final Silica Concentrate')
            plt.xlabel('Raw % Silica Feed')
            plt.ylabel('Final % Silica Concentrate')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.savefig('static_scatterplot.png', dpi=150)
            plt.close()

            # --- ANIMATED PLOTS ---
            print("   Compiling individual trend animations (fixing layout spacing)...")
            
            step = max(1, len(self.df) // 200)
            anim_dates = dates[::step]
            anim_silica_feed = silica_feed[::step]
            anim_silica_conc = silica_concentrate[::step]

            # Animation 1: Raw Silica Feed
            fig1, ax1 = plt.subplots(figsize=(8, 4.5))
            line1, = ax1.plot([], [], color='darkred', lw=2, label='Raw % Silica Feed')
            ax1.set_title('Animated Raw Silica Feed Trend Over Time', pad=15)
            ax1.set_ylabel('Feed %')
            ax1.set_xlim(anim_dates[0], anim_dates[-1])
            ax1.set_ylim(np.min(anim_silica_feed) - 0.5, np.max(anim_silica_feed) + 0.5)
            ax1.grid(True, linestyle=':', alpha=0.5)
            ax1.tick_params(axis='x', rotation=15)
            ax1.legend(loc='upper right')
            plt.tight_layout()

            def init1():
                line1.set_data([], [])
                return line1,

            def update1(frame):
                line1.set_data(anim_dates[:frame], anim_silica_feed[:frame])
                return line1,

            ani1 = animation.FuncAnimation(fig1, update1, frames=len(anim_dates), init_func=init1, blit=True, interval=50)
            ani1.save('animation_silica_feed.gif', writer='pillow')
            plt.close()

            # Animation 2: Final Silica Concentrate
            fig2, ax2 = plt.subplots(figsize=(8, 4.5))
            line2, = ax2.plot([], [], color='purple', lw=2, label='Final % Silica Concentrate')
            ax2.set_title('Animated Final Impurity Concentrate Over Time', pad=15)
            ax2.set_ylabel('Concentrate %')
            ax2.set_xlim(anim_dates[0], anim_dates[-1])
            ax2.set_ylim(np.min(anim_silica_conc) - 0.5, np.max(anim_silica_conc) + 0.5)
            ax2.grid(True, linestyle=':', alpha=0.5)
            ax2.tick_params(axis='x', rotation=15)
            ax2.legend(loc='upper right')
            plt.tight_layout()

            def init2():
                line2.set_data([], [])
                return line2,

            def update2(frame):
                line2.set_data(anim_dates[:frame], anim_silica_conc[:frame])
                return line2,

            ani2 = animation.FuncAnimation(fig2, update2, frames=len(anim_dates), init_func=init2, blit=True, interval=50)
            ani2.save('animation_silica_concentrate.gif', writer='pillow')
            plt.close()

            print("   Success: All visualizations rendered and saved completely!")
            print("\nALL PIPELINE STAGES COMPLETED EXCELLENTLY!")

        except Exception as e:
            print(f"Error: An error occurred during visualization rendering: {e}")

if __name__ == "__main__":
    pipeline = MiningDataPipeline("data/datasets_original.csv")
    
    pipeline.ingest_data()
    pipeline.clean_data(month_filter="2017-06")
    pipeline.analyze_statistics()
    pipeline.generate_visualizations()