import logging
from pathlib import Path
import sys
from pandorascheduler_rework import science_calendar

def regenerate_calendar(output_dir_str: str):
    output_dir = Path(output_dir_str).resolve()
    # Find the schedule CSV
    schedule_csv = list(output_dir.glob("Pandora_Schedule_*.csv"))[0]
    data_dir = output_dir / "data"
    
    print("Regenerating Science Calendar...")
    print(f"  Schedule: {schedule_csv}")
    print(f"  Data Dir: {data_dir}")
    print(f"  Output:   {output_dir}")

    if not data_dir.exists():
        print(f"ERROR: Data directory not found at {data_dir}")
        sys.exit(1)

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("regenerate")

    inputs = science_calendar.ScienceCalendarInputs(
        schedule_csv=schedule_csv,
        data_dir=data_dir,
    )

    config = science_calendar.ScienceCalendarConfig(
        visit_limit=None,  # Generate all visits
        prioritise_occultations_by_slew=False,
        author="Pandora Scheduler",
    )

    output_path = output_dir / "Pandora_science_calendar.xml"

    science_calendar.generate_science_calendar(
        inputs,
        output_path=output_path,
        config=config,
    )
    print(f"SUCCESS: Generated {output_path}")

if __name__ == "__main__":
    regenerate_calendar("output_standalone")
