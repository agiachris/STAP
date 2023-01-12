(define (problem structured_language_0)
	(:domain geometric_workspace)
	(:objects
		hook - tool
		red_box - box
	)
	(:init
		(on hook table)
		(on red_box table)
        ; Geometric facts
        (inworkspace table)
        (inworkspace hook)
        (beyondworkspace red_box)
	)
	(:goal (and
		(inhand red_box)
	))
)
