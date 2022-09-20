(define (domain workspace)
	(:requirements :strips :typing :equality :universal-preconditions :negative-preconditions :conditional-effects)
	(:types
		physobj - object
		movable - physobj
	)
	(:constants table - physobj)
	(:predicates
		(inhand ?a - movable)
		(on ?a - movable ?b - physobj)
		(inworkspace ?a - movable)
	)
	(:action pick
		:parameters (?a - movable ?b - physobj)
		:precondition (and
			(on ?a ?b)  ; TODO: Remove
			(forall (?b - movable)
				(and
					(not (inhand ?b))
					(not (on ?b ?a))
				)
			)
		)
		:effect (and
			(inhand ?a)
			(forall (?b - physobj) (not (on ?a ?b)))
		)
	)
	(:action place
		:parameters (?a - movable ?b - physobj)
		:precondition (and
			(not (= ?a ?b))
			(inhand ?a)
		)
		:effect (and
			(not (inhand ?a))
			(on ?a ?b)
		)
	)
	(:action pull
		:parameters (?a - movable ?b - movable)
		:precondition (and
			(not (= ?a ?b))
			(inhand ?b)
			(on ?a table)
		)
		:effect (and
			(inworkspace ?a)
		)
	)
)
