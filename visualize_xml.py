from scripts.visualizer import ScheduleVisualizer
from pandorascheduler_rework.models import ScienceCalendar, Visit, Sequence
from astropy.time import Time
import matplotlib.pyplot as plt

from scripts.parser import parse_science_calendar

class DummyScheduler:
    def get_gap_report(self):
        return {}

def main():
    # Parse calendar
    calendar = parse_science_calendar('/Users/vkostov/Documents/GitHub/pandora-scheduler/output_directory/data/Pandora_science_calendar.xml')

    # Create a dummy scheduler
    dummy_scheduler = DummyScheduler()

    # Create a ScheduleVisualizer instance with the dummy scheduler
    visualizer = ScheduleVisualizer(dummy_scheduler)

    # Call the method
    figures = visualizer.plot_gantt_timeline_by_priority_windowed(calendar,figsize=(20, 10),show_sequence_labels=True,title="Sample Schedule by Priority",window_days=4)

    # Save or display the figures
    for i, fig in enumerate(figures):
        fig.savefig(f"/Users/vkostov/Documents/GitHub/pandora-scheduler/output_directory/confirm_visibility/gantt_plot_window_{i}.png")
        plt.close(fig)

if __name__ == "__main__":
    main()