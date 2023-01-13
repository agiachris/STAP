(define (problem structured_language_0)
	(:domain geometric_workspace)
	(:objects
        rack - receptacle
		hook - tool
		red_box - box
	)
	(:init
		(on rack table)
		(on hook table)
		(on red_box table)
        ; Geometric facts
        (inworkspace table)
        (inworkspace rack)
        (inworkspace hook)
        (beyondworkspace red_box)
	)
	(:goal (and
		(inhand red_box)
	))
)
