(define (problem structured_language_2)
	(:domain geometric_workspace)
	(:objects
        rack - receptacle
		hook - tool
        yellow_box - box
        cyan_box - box
	)
	(:init
		(on rack table)
        (on hook table)
		(on yellow_box table)
		(on cyan_box table)
        ; Geometric facts
        (inworkspace table)
        (inworkspace hook)
        (inoperationalzone yellow_box)
        (inobstructionzone cyan_box)
        (beyondworkspace rack)
        (infront yellow_box rack)
        (infront cyan_box rack)
        (nonblocking rack hook)
	)
	(:goal (and
        (under yellow_box rack)
	))
)
